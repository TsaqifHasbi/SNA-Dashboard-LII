"""
Training Data Labeling & Stratified Split for IndoBERT (3 Sentimen)
Mempersiapkan dataset latih 3 kelas: Negatif (0), Netral (1), Positif (2)
Mengintegrasikan Gold Standard Annotation (jika tersedia) + Contextual Guardrail Scorer.
"""

import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
MASTER_PARQUET = os.path.join(OUTPUT_DIR, 'master_scraped_corpus.parquet')
MASTER_CSV = os.path.join(OUTPUT_DIR, 'master_scraped_corpus.csv')
GOLD_PATH = os.path.join(OUTPUT_DIR, 'gold_standard_annotated_1000.csv')

POSITIVE_LEXICON = {
    'bagus', 'hebat', 'mantap', 'keren', 'berhasil', 'sukses', 'bermanfaat', 'membantu',
    'apresiasi', 'dukung', 'mendukung', 'terima kasih', 'alhamdulillah', 'bersyukur', 'berkah',
    'berkualitas', 'amanah', 'bangga', 'solid', 'kompak', 'khidmat', 'peduli', 'santun',
    'damai', 'sejuk', 'rukun', 'inspiratif', 'maju', 'tumbuh', 'harapan', 'terbukti', 'layak',
    'adil', 'merata', 'beres', 'tuntas', 'positif', 'puas', 'juara', 'menyejahterakan',
    'terobosan', 'efisien', 'transparan', 'semangat', 'berdaya', 'maslahat', 'istiqomah'
}

NEGATIVE_LEXICON = {
    'buruk', 'gagal', 'rusak', 'ambrol', 'hancur', 'kecewa', 'rugi', 'korupsi',
    'bancakan', 'curang', 'bohong', 'palsu', 'maling', 'konflik', 'pecah', 'ricuh',
    'kisruh', 'polemik', 'kontroversi', 'parah', 'krisis', 'marah', 'sesat', 'bobrok',
    'bodoh', 'sengsara', 'beban', 'masalah', 'cacat', 'pencitraan', 'rugikan',
    'penipuan', 'manipulasi', 'menolak', 'tolak', 'kritik', 'protes', 'salah',
    'amburadul', 'merusak', 'gaduh', 'bubar', 'bubarkan', 'penyelewengan', 'politisasi'
}

NEGATION_WORDS = {'tidak', 'tak', 'bukan', 'jangan', 'belum', 'gak', 'ga', 'nggak', 'tdk', 'kurang'}

try:
    from src.sentiment_corrector import correct_sentiment
except Exception:
    correct_sentiment = None


def rule_based_sentiment_scorer(text):
    """
    Menghitung skor sentimen berbasis leksikon kontekstual, negasi, & guardrail sarkasme.
    Return: 0 (Negatif), 1 (Netral), 2 (Positif)
    """
    if not isinstance(text, str) or not text.strip():
        return 1

    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 1

    pos_score = 0.0
    neg_score = 0.0
    for i, w in enumerate(words):
        is_neg = (i > 0 and words[i - 1] in NEGATION_WORDS) or (i > 1 and words[i - 2] in NEGATION_WORDS)
        if w in POSITIVE_LEXICON:
            if is_neg:
                neg_score += 1.5
            else:
                pos_score += 1.0
        elif w in NEGATIVE_LEXICON:
            if is_neg:
                pos_score += 0.8
            else:
                neg_score += 1.5

    diff = pos_score - neg_score
    if diff >= 1.0:
        raw_sent, raw_pol = 'Positif', 0.65
    elif diff <= -1.0:
        raw_sent, raw_pol = 'Negatif', -0.65
    else:
        raw_sent, raw_pol = 'Netral', 0.0

    if correct_sentiment is not None:
        try:
            corr_sent, _, _, reason = correct_sentiment(text, raw_sent, raw_pol)
            if reason:
                raw_sent = corr_sent
        except Exception:
            pass

    return {'Negatif': 0, 'Netral': 1, 'Positif': 2}.get(raw_sent, 1)


def prepare_labeled_dataset(sample_size=30000):
    """Menyiapkan train.csv, val.csv, test.csv terstratifikasi untuk fine-tuning IndoBERT 3 Sentimen."""
    print("\n" + "=" * 75)
    print("🏷️  MEMPERSIAPKAN DATASET ANOTASI TERSTRATIFIKASI UNTUK INDOBERT (3 SENTIMEN)")
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
        raise RuntimeError("Master corpus kosong. Jalankan scraper terlebih dahulu.")

    df = df.dropna(subset=['cleaned_content']).copy()
    df['cleaned_content'] = df['cleaned_content'].astype(str)
    df = df[df['cleaned_content'].str.strip().str.len() >= 5].reset_index(drop=True)
    print(f"Total Master Corpus Valid: {len(df):,} baris")

    df['label'] = df['cleaned_content'].map(rule_based_sentiment_scorer)

    # Integrasikan Gold Standard jika tersedia
    if os.path.exists(GOLD_PATH):
        try:
            gold_df = pd.read_csv(GOLD_PATH)
            if 'cleaned_content' in gold_df.columns and 'sentiment_label' in gold_df.columns:
                gold_map = dict(zip(gold_df['cleaned_content'].astype(str), gold_df['sentiment_label'].astype(int)))
                matched = df['cleaned_content'].isin(gold_map)
                df.loc[matched, 'label'] = df.loc[matched, 'cleaned_content'].map(gold_map)
                print(f"   ✓ Mengintegrasikan {int(matched.sum()):,} label terverifikasi dari Gold Standard!")
        except Exception as e:
            print(f"   [WARN] Gagal memuat Gold Standard: {e}")

    label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    df['sentiment'] = df['label'].map(label_map)

    print("\nDistribusi Sentimen pada Seluruh Corpus:")
    print(df['sentiment'].value_counts().to_string())

    n_topics = max(1, df['topic'].nunique()) if 'topic' in df.columns else 1
    target_per_group = max(10, int(sample_size / (n_topics * 3)))
    sample_indices = []
    group_cols = ['topic', 'label'] if 'topic' in df.columns else ['label']
    for _, group in df.groupby(group_cols):
        n_take = min(len(group), target_per_group)
        if n_take > 0:
            sample_indices.extend(group.sample(n_take, random_state=42).index.tolist())

    sample_df = df.loc[sample_indices].sample(frac=1.0, random_state=42).copy().reset_index(drop=True)
    print(f"\nUkuran Dataset Training IndoBERT Terpilih: {len(sample_df):,} baris")
    print(sample_df['sentiment'].value_counts().to_string())

    # Pastikan tiap kelas punya minimal 4 sampel agar stratified split 80/10/10 aman
    vc = sample_df['label'].value_counts()
    can_stratify = (len(vc) >= 2 and vc.min() >= 4)

    train_df, temp_df = train_test_split(
        sample_df, test_size=0.20, random_state=42,
        stratify=sample_df['label'] if can_stratify else None
    )
    vc_temp = temp_df['label'].value_counts()
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42,
        stratify=temp_df['label'] if (len(vc_temp) >= 2 and vc_temp.min() >= 2) else None
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False, encoding='utf-8-sig')
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False, encoding='utf-8-sig')
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False, encoding='utf-8-sig')

    print(f"\n📦 Pembagian Dataset Tersimpan di {OUTPUT_DIR}:")
    print(f"   - Train Set (80%)      : {len(train_df):,} baris")
    print(f"   - Validation Set (10%) : {len(val_df):,} baris")
    print(f"   - Test Set (10%)       : {len(test_df):,} baris")
    return train_df, val_df, test_df


if __name__ == '__main__':
    prepare_labeled_dataset()
