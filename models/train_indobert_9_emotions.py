"""
Fine-Tuning IndoBERT untuk 9 Kelas Emosi (Plutchik Standard)
Model Base: indobenchmark/indobert-base-p1
Optimasi: Dynamic Padding (DataCollatorWithPadding) + Mixed Precision (AMP CUDA)
Output: models/best_indobert_emotion (sinkron langsung dengan src/indobert_engine.py & app.py)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'best_indobert_emotion')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_NAME = "indobenchmark/indobert-base-p1"
NUM_LABELS = 9

EMOTION_9_CLASSES = [
    'Trust', 'Anticipation', 'Joy',
    'Disgust', 'Fear', 'Sadness', 'Anger',
    'Surprise', 'Neutral',
]
LABEL2ID = {name: idx for idx, name in enumerate(EMOTION_9_CLASSES)}
ID2LABEL = {idx: name for idx, name in enumerate(EMOTION_9_CLASSES)}
EMOTION_TO_SENTIMENT = {
    'Trust': 'Positif', 'Anticipation': 'Positif', 'Joy': 'Positif',
    'Disgust': 'Negatif', 'Fear': 'Negatif', 'Sadness': 'Negatif', 'Anger': 'Negatif',
    'Surprise': 'Netral', 'Neutral': 'Netral',
}


def check_dependencies():
    missing = []
    for pkg, mod in [('torch', 'torch'), ('transformers', 'transformers'), ('scikit-learn', 'sklearn')]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"⚠️ Library belum lengkap. Jalankan: pip install {' '.join(missing)}")
        return False
    return True


def run_training_pipeline(epochs=4, batch_size=16, learning_rate=2e-5, max_length=128):
    if not check_dependencies():
        return

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )
    from torch.optim import AdamW
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"\n🚀 Device Training IndoBERT 9-Emosi: {device}")

    train_file = os.path.join(DATA_DIR, 'train_emotion_9.csv')
    val_file = os.path.join(DATA_DIR, 'val_emotion_9.csv')
    test_file = os.path.join(DATA_DIR, 'test_emotion_9.csv')

    if not os.path.exists(train_file):
        from models.prepare_9_emotions_dataset import load_and_standardize_corpus, generate_balanced_splits
        df_c = load_and_standardize_corpus()
        generate_balanced_splits(df_c)

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class EmotionDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                [str(t) for t in texts],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.labels = [int(l) for l in labels]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.labels[idx],
            }

    train_loader = DataLoader(EmotionDataset(train_df['cleaned_content'], train_df['label']), batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(EmotionDataset(val_df['cleaned_content'], val_df['label']), batch_size=batch_size * 2, collate_fn=collator)
    test_loader = DataLoader(EmotionDataset(test_df['cleaned_content'], test_df['label']), batch_size=batch_size * 2, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    best_f1 = -1.0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(f"  Ep {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        _, _, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='macro', zero_division=0)
        print(f"  >> Epoch {epoch+1} ({time.time()-t0:.0f}s): Val Acc={val_acc*100:.2f}% | Val Macro-F1={val_f1*100:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"  ⭐ Checkpoint Terbaik Disimpan ke {MODEL_SAVE_DIR}!")

    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(device)
    best_model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            test_targets.extend(batch['labels'].numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
    _, _, test_wf1, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(NUM_LABELS)))

    pred_sentiments = [EMOTION_TO_SENTIMENT[ID2LABEL[p]] for p in test_preds]
    true_sentiments = [EMOTION_TO_SENTIMENT[ID2LABEL[t]] for t in test_targets]
    sent_acc = accuracy_score(true_sentiments, pred_sentiments)

    print(f"\n🎯 TEST SET 9-EMOSI: Acc={test_acc*100:.2f}% | Macro-F1={test_f1*100:.2f}% | Implicit Sentiment Acc={sent_acc*100:.2f}%")
    print(classification_report(test_targets, test_preds, labels=list(range(NUM_LABELS)), target_names=EMOTION_9_CLASSES, zero_division=0))

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_9_CLASSES, yticklabels=EMOTION_9_CLASSES, ax=ax)
        ax.set_title(f"Confusion Matrix IndoBERT 9 Emosi (Acc = {test_acc*100:.2f}%)", fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrix_9_emotions.png'), dpi=300)
        plt.close()
    except Exception:
        pass

    metrics = {
        'model_name': MODEL_NAME,
        'task': 'emotion_classification_9_classes',
        'classes': EMOTION_9_CLASSES,
        'label_names': EMOTION_9_CLASSES,
        'label2id': LABEL2ID,
        'id2label': ID2LABEL,
        'emotion_to_sentiment': EMOTION_TO_SENTIMENT,
        'accuracy': float(test_acc),
        'test_accuracy': float(test_acc),
        'macro_precision': float(test_prec),
        'macro_recall': float(test_rec),
        'macro_f1': float(test_f1),
        'weighted_f1': float(test_wf1),
        'implicit_sentiment_accuracy': float(sent_acc),
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(MODEL_SAVE_DIR, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"📝 Metrik evaluasi tersimpan di: {os.path.join(MODEL_SAVE_DIR, 'evaluation_metrics.json')}")


if __name__ == '__main__':
    run_training_pipeline()
