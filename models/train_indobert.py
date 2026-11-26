"""
IndoBERT Fine-Tuning Pipeline for 3-Class Sentiment Analysis
Model Base: indobenchmark/indobert-base-p1
Optimasi: Dynamic Padding (DataCollatorWithPadding) + Mixed Precision (AMP CUDA)
Output: models/best_indobert_sentiment + evaluation_metrics.json + confusion_matrix.png
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
MODEL_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'best_indobert_sentiment')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_NAME = "indobenchmark/indobert-base-p1"
NUM_LABELS = 3
LABEL_NAMES = ['Negatif', 'Netral', 'Positif']
LABEL2ID = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
ID2LABEL = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}


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


def run_training_pipeline(epochs=3, batch_size=16, learning_rate=2e-5, max_length=128):
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
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"\n🚀 Device Training IndoBERT 3-Sentimen: {device}")
    if use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name(0)} (AMP Mixed Precision Aktif)")

    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')

    if not os.path.exists(train_path):
        from models.label_training_data import prepare_labeled_dataset
        train_df, val_df, test_df = prepare_labeled_dataset()
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class SentimentDataset(Dataset):
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

    train_loader = DataLoader(SentimentDataset(train_df['cleaned_content'], train_df['label']), batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(SentimentDataset(val_df['cleaned_content'], val_df['label']), batch_size=batch_size * 2, collate_fn=collator)
    test_loader = DataLoader(SentimentDataset(test_df['cleaned_content'], test_df['label']), batch_size=batch_size * 2, collate_fn=collator)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    best_val_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            total_train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"   Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f} | {time.time()-t0:.1f}s")

        model.eval()
        val_preds, val_trues = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()
                val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_trues, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_trues, val_preds, average='macro', zero_division=0)
        print(f"📈 [Epoch {epoch+1}] Val Acc={val_acc*100:.2f}% | Val Macro-F1={val_f1*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"   💾 Checkpoint terbaik disimpan ke {MODEL_SAVE_DIR}")

    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(device)
    best_model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
            test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            test_trues.extend(batch['labels'].numpy())

    acc = accuracy_score(test_trues, test_preds)
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(test_trues, test_preds, average='macro', zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(test_trues, test_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(test_trues, test_preds, labels=[0, 1, 2])

    print(f"\n📊 TEST SET 3-SENTIMEN: Acc={acc*100:.2f}% | Macro-F1={macro_f1*100:.2f}% | Weighted-F1={weighted_f1*100:.2f}%")
    print(classification_report(test_trues, test_preds, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0))

    try:
        import seaborn as sns
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.title(f"Confusion Matrix IndoBERT Sentimen (Acc: {acc*100:.2f}%)", fontsize=12, fontweight='bold')
        plt.xlabel("Prediksi Model")
        plt.ylabel("Aktual (Ground Truth)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
    except Exception:
        pass

    metrics_summary = {
        'model_name': MODEL_NAME,
        'accuracy': float(acc),
        'test_accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': cm.tolist(),
        'label_names': LABEL_NAMES,
    }
    with open(os.path.join(MODEL_SAVE_DIR, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✅ Evaluasi tersimpan di {os.path.join(MODEL_SAVE_DIR, 'evaluation_metrics.json')}")


if __name__ == '__main__':
    run_training_pipeline()
