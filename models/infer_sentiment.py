"""
IndoBERT Dual-Model Batch Inference (3 Sentimen + 9 Emosi) & Guardrail Enrichment
Menghasilkan data/enriched_corpus_for_sna.parquet & .csv yang langsung siap dipakai
oleh Streamlit Dashboard (Format Laporan Harian & Cockpit Evaluasi) serta SNA Engine.
"""

import os
import sys
import pandas as pd
import numpy as np

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
MASTER_PARQUET = os.path.join(DATA_DIR, 'master_scraped_corpus.parquet')
MASTER_CSV = os.path.join(DATA_DIR, 'master_scraped_corpus.csv')

SENT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_indobert_sentiment')
EMO_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'best_indobert_emotion')

OUTPUT_PARQUET = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
OUTPUT_CSV = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.csv')

SENT_ID2LABEL = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
SENT_LABEL2ID = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
EMO_ID2LABEL = {
    0: 'Trust', 1: 'Anticipation', 2: 'Joy',
    3: 'Disgust', 4: 'Fear', 5: 'Sadness', 6: 'Anger',
    7: 'Surprise', 8: 'Neutral',
}


def batch_infer_sentiment(batch_size=64, max_length=128):
    print("=" * 75)
    print("🔮 MEMULAI DUAL-MODEL BATCH INFERENCE (3 SENTIMEN + 9 EMOSI + GUARDRAIL)")
    print("=" * 75)

    if os.path.exists(MASTER_PARQUET):
        df = pd.read_parquet(MASTER_PARQUET)
    elif os.path.exists(MASTER_CSV):
        df = pd.read_csv(MASTER_CSV, low_memory=False)
    else:
        print("[!] Master corpus belum ada. Menjalankan konsolidasi terlebih dahulu...")
        from models.dataset_preparation import consolidate_all_platforms
        df = consolidate_all_platforms()

    if df is None or df.empty:
        print("[WARN] Tidak ada data untuk di-enrich.")
        return pd.DataFrame()

    df['cleaned_content'] = df['cleaned_content'].fillna('').astype(str)
    if 'raw_content' not in df.columns and 'content' in df.columns:
        df['raw_content'] = df['content']
    texts = df['cleaned_content'].tolist()
    n_rows = len(texts)
    print(f"Total Dokumen yang Akan Di-Enrich: {n_rows:,} baris")

    has_sent_model = os.path.exists(os.path.join(SENT_MODEL_DIR, 'config.json'))
    has_emo_model = os.path.exists(os.path.join(EMO_MODEL_DIR, 'config.json'))

    raw_sents = None
    raw_scores = None
    raw_emos = None

    if has_sent_model or has_emo_model:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            if has_sent_model:
                print(f"   ✓ Memuat IndoBERT 3-Sentimen dari {SENT_MODEL_DIR} ({device})...")
                tok_s = AutoTokenizer.from_pretrained(SENT_MODEL_DIR)
                mod_s = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_DIR).to(device)
                mod_s.eval()
                preds_s, scores_s = [], []
                with torch.no_grad():
                    for start in range(0, n_rows, batch_size):
                        batch_t = texts[start:start + batch_size]
                        enc = tok_s(batch_t, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
                        with torch.amp.autocast('cuda', enabled=use_cuda):
                            logits = mod_s(**enc).logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        preds_s.extend([SENT_ID2LABEL.get(int(i), 'Netral') for i in np.argmax(probs, axis=1)])
                        scores_s.extend((probs[:, 2] - probs[:, 0]).tolist())
                raw_sents = preds_s
                raw_scores = scores_s

            if has_emo_model:
                print(f"   ✓ Memuat IndoBERT 9-Emosi dari {EMO_MODEL_DIR} ({device})...")
                tok_e = AutoTokenizer.from_pretrained(EMO_MODEL_DIR)
                mod_e = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL_DIR).to(device)
                mod_e.eval()
                preds_e = []
                with torch.no_grad():
                    for start in range(0, n_rows, batch_size):
                        batch_t = texts[start:start + batch_size]
                        enc = tok_e(batch_t, truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
                        with torch.amp.autocast('cuda', enabled=use_cuda):
                            logits = mod_e(**enc).logits
                        probs = torch.softmax(logits, dim=1).cpu().numpy()
                        preds_e.extend([EMO_ID2LABEL.get(int(i), 'Neutral') for i in np.argmax(probs, axis=1)])
                raw_emos = preds_e
        except Exception as e:
            print(f"[WARN] Neural batch inference gagal ({e}). Beralih ke Calibrated Contextual Engine...")

    if raw_sents is None:
        print("   [INFO] Menggunakan Calibrated Contextual Sentiment Scorer...")
        from models.label_training_data import rule_based_sentiment_scorer
        lbls = [rule_based_sentiment_scorer(t) for t in texts]
        raw_sents = [SENT_ID2LABEL.get(i, 'Netral') for i in lbls]
        score_map = {'Negatif': -0.72, 'Netral': 0.0, 'Positif': 0.72}
        raw_scores = [score_map[s] for s in raw_sents]

    if raw_emos is None:
        print("   [INFO] Menggunakan Plutchik 9-Emotion Contextual Detector...")
        from src.data_loader import detect_emotion
        raw_emos = [detect_emotion(t) for t in texts]

    # Guardrail Post-Processing (Sarkasme, Profanity, Kritik/Apresiasi, & Valence Alignment)
    from src.sentiment_corrector import correct_sentiment, align_emotion_valence_9class
    final_sents, final_scores, final_emos = [], [], []
    guardrail_fixes = 0

    for txt, s_in, sc_in, e_in in zip(texts, raw_sents, raw_scores, raw_emos):
        s_corr, _, p_corr, reason = correct_sentiment(txt, s_in, float(sc_in))
        if reason:
            s_out, p_out = s_corr, float(p_corr)
            guardrail_fixes += 1
        else:
            s_out, p_out = s_in, float(sc_in)
        e_out = align_emotion_valence_9class(txt, s_out, e_in)
        final_sents.append(s_out)
        final_scores.append(round(p_out, 4))
        final_emos.append(e_out)

    df['predicted_sentiment'] = final_sents
    df['sentiment_name'] = final_sents
    df['sentiment_label'] = [SENT_LABEL2ID.get(s, 1) for s in final_sents]
    df['sentiment_score'] = final_scores
    df['emotion'] = final_emos
    df['emotion_9'] = final_emos

    pos_count = int((df['sentiment_name'] == 'Positif').sum())
    neg_count = int((df['sentiment_name'] == 'Negatif').sum())
    net_count = int((df['sentiment_name'] == 'Netral').sum())
    nss = ((pos_count - neg_count) / max(1, n_rows)) * 100.0

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 75)
    print("📊 HASIL ENRICHMENT KORPUS MULTI-PLATFORM:")
    print(f"   • Total Dokumen       : {n_rows:,}")
    print(f"   • Positif             : {pos_count:>8,} ({pos_count/n_rows*100:.1f}%)")
    print(f"   • Netral              : {net_count:>8,} ({net_count/n_rows*100:.1f}%)")
    print(f"   • Negatif             : {neg_count:>8,} ({neg_count/n_rows*100:.1f}%)")
    print(f"   • Net Sentiment (NSS) : {nss:+.1f}%")
    print(f"   • Guardrail Corrected : {guardrail_fixes:,} baris sarkasme/anomali dikoreksi")
    print(f"   • Parquet             : {OUTPUT_PARQUET}")
    print(f"   • CSV                 : {OUTPUT_CSV}")
    print("=" * 75)
    return df


if __name__ == '__main__':
    batch_infer_sentiment()
