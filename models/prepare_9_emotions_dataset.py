"""
Script Generator Dataset 9 Emosi (Plutchik Standard) untuk Fine-Tuning IndoBERT
Struktur 9 Emosi:
- Positif (3): Trust, Anticipation, Joy
- Negatif (4): Disgust, Fear, Sadness, Anger
- Netral  (2): Surprise, Neutral
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
ENRICHED_PARQUET = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
MASTER_PARQUET = os.path.join(DATA_DIR, 'master_scraped_corpus.parquet')
MASTER_CSV = os.path.join(DATA_DIR, 'master_scraped_corpus.csv')

EMOTION_9_CLASSES = [
    'Trust',         # 0: Positif
    'Anticipation',  # 1: Positif
    'Joy',           # 2: Positif
    'Disgust',       # 3: Negatif
    'Fear',          # 4: Negatif
    'Sadness',       # 5: Negatif
    'Anger',         # 6: Negatif
    'Surprise',      # 7: Netral
    'Neutral',       # 8: Netral
]

LABEL2ID = {name: idx for idx, name in enumerate(EMOTION_9_CLASSES)}
ID2LABEL = {idx: name for idx, name in enumerate(EMOTION_9_CLASSES)}

EMOTION_TO_SENTIMENT = {
    'Trust': 'Positif', 'Anticipation': 'Positif', 'Joy': 'Positif',
    'Disgust': 'Negatif', 'Fear': 'Negatif', 'Sadness': 'Negatif', 'Anger': 'Negatif',
    'Surprise': 'Netral', 'Neutral': 'Netral',
}

STANDARDIZE_MAP = {
    'Senang': 'Joy', 'Joy': 'Joy',
    'Percaya': 'Trust', 'Trust': 'Trust',
    'Antisipasi': 'Anticipation', 'Anticipation': 'Anticipation',
    'Jijik': 'Disgust', 'Disgust': 'Disgust',
    'Takut': 'Fear', 'Fear': 'Fear',
    'Sedih': 'Sadness', 'Sadness': 'Sadness',
    'Marah': 'Anger', 'Anger': 'Anger',
    'Terkejut': 'Surprise', 'Surprise': 'Surprise',
    'Informatif': 'Neutral', 'Netral': 'Neutral', 'Neutral': 'Neutral',
}


def load_and_standardize_corpus():
    if os.path.exists(ENRICHED_PARQUET):
        print(f"Memuat korpus dari {ENRICHED_PARQUET}...")
        df = pd.read_parquet(ENRICHED_PARQUET)
    elif os.path.exists(MASTER_PARQUET):
        print(f"Memuat korpus dari {MASTER_PARQUET} & menjalankan deteksi 9 emosi awal...")
        df = pd.read_parquet(MASTER_PARQUET)
    elif os.path.exists(MASTER_CSV):
        print(f"Memuat korpus dari {MASTER_CSV} & menjalankan deteksi 9 emosi awal...")
        df = pd.read_csv(MASTER_CSV, low_memory=False)
    else:
        from models.dataset_preparation import consolidate_all_platforms
        df = consolidate_all_platforms()

    df = df.dropna(subset=['cleaned_content']).copy()
    df['cleaned_content'] = df['cleaned_content'].astype(str)
    df = df[df['cleaned_content'].str.strip().str.len() > 8].reset_index(drop=True)

    if 'emotion_9' in df.columns:
        df['emotion_9'] = df['emotion_9'].replace(STANDARDIZE_MAP)
    elif 'emotion' in df.columns:
        df['emotion_9'] = df['emotion'].replace(STANDARDIZE_MAP)
    else:
        from src.data_loader import detect_emotion
        from models.label_training_data import rule_based_sentiment_scorer
        from src.sentiment_corrector import align_emotion_valence_9class
        s_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sents = df['cleaned_content'].map(rule_based_sentiment_scorer).map(s_map)
        raw_emos = df['cleaned_content'].map(detect_emotion)
        df['emotion_9'] = [
            align_emotion_valence_9class(t, s, e)
            for t, s, e in zip(df['cleaned_content'], sents, raw_emos)
        ]

    df = df[df['emotion_9'].isin(EMOTION_9_CLASSES)].copy()
    print(f"Total baris valid untuk 9 Emosi: {len(df):,}")
    vc = df['emotion_9'].value_counts()
    for emo in EMOTION_9_CLASSES:
        print(f"  [{EMOTION_TO_SENTIMENT[emo].upper():7s}] {emo:12s}: {vc.get(emo, 0):>7,} baris")
    return df


def generate_balanced_splits(df, samples_per_class=3500, random_state=42):
    print(f"\nMelakukan Stratified Balanced Sampling (target maks {samples_per_class:,}/kelas)...")
    sampled_dfs = []
    for emo in EMOTION_9_CLASSES:
        subset = df[df['emotion_9'] == emo]
        n_avail = len(subset)
        if n_avail == 0:
            continue
        n_take_total = min(samples_per_class, n_avail)
        max_per_platform = max(1, int(n_take_total * 0.45))
        subs = []
        if 'platform' in subset.columns:
            for plat in subset['platform'].unique():
                p_match = subset[subset['platform'] == plat]
                p_take = min(len(p_match), max_per_platform)
                if p_take > 0:
                    subs.append(p_match.sample(n=p_take, random_state=random_state))
            comb = pd.concat(subs) if subs else subset.sample(n=n_take_total, random_state=random_state)
        else:
            comb = subset.sample(n=n_take_total, random_state=random_state)

        if len(comb) > n_take_total:
            comb = comb.sample(n=n_take_total, random_state=random_state)
        elif len(comb) < n_take_total:
            rem_pool = subset[~subset.index.isin(comb.index)]
            if len(rem_pool) > 0:
                comb = pd.concat([comb, rem_pool.sample(n=min(n_take_total - len(comb), len(rem_pool)), random_state=random_state)])
        sampled_dfs.append(comb)

    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    balanced_df['label'] = balanced_df['emotion_9'].map(LABEL2ID)
    balanced_df['sentiment'] = balanced_df['emotion_9'].map(EMOTION_TO_SENTIMENT)
    if 'platform' not in balanced_df.columns:
        balanced_df['platform'] = 'SocialMedia'
    if 'topic' not in balanced_df.columns:
        balanced_df['topic'] = 'General'

    vc = balanced_df['label'].value_counts()
    can_strat = (len(vc) >= 2 and vc.min() >= 4)
    train_df, temp_df = train_test_split(
        balanced_df, test_size=0.20, random_state=random_state,
        stratify=balanced_df['label'] if can_strat else None
    )
    vc_t = temp_df['label'].value_counts()
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state,
        stratify=temp_df['label'] if (len(vc_t) >= 2 and vc_t.min() >= 2) else None
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    cols_to_save = ['cleaned_content', 'emotion_9', 'label', 'sentiment', 'platform', 'topic']
    train_df[cols_to_save].to_csv(os.path.join(DATA_DIR, 'train_emotion_9.csv'), index=False, encoding='utf-8')
    val_df[cols_to_save].to_csv(os.path.join(DATA_DIR, 'val_emotion_9.csv'), index=False, encoding='utf-8')
    test_df[cols_to_save].to_csv(os.path.join(DATA_DIR, 'test_emotion_9.csv'), index=False, encoding='utf-8')

    meta_path = os.path.join(DATA_DIR, 'emotion_9_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'classes': EMOTION_9_CLASSES,
            'label2id': LABEL2ID,
            'id2label': ID2LABEL,
            'emotion_to_sentiment': EMOTION_TO_SENTIMENT,
            'samples_per_class': samples_per_class,
            'total_train': len(train_df),
            'total_val': len(val_df),
            'total_test': len(test_df),
        }, f, indent=2)

    print(f"\n✅ Dataset 9 Emosi tersimpan di {DATA_DIR} (Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,})")
    return train_df, val_df, test_df


if __name__ == '__main__':
    df_corp = load_and_standardize_corpus()
    generate_balanced_splits(df_corp, samples_per_class=3500)
