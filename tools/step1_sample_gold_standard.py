"""
Langkah 1: Stratified Sampling 1.000 Sampel untuk Gold Standard Annotation
Sampling proporsional dari seluruh 7 platform, topik, dan sentimen (termasuk kasus sarkasme & negasi).
"""
import os
import sys
import pandas as pd
import numpy as np

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ENRICHED_PARQUET = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
MASTER_PARQUET = os.path.join(DATA_DIR, 'master_scraped_corpus.parquet')
OUTPUT_PATH = os.path.join(DATA_DIR, 'gold_standard_sample_1000.csv')


def main(target_total=1000):
    print("=" * 70)
    print(f"📊 STRATIFIED SAMPLING {target_total:,} SAMPEL UNTUK GOLD STANDARD (7 PLATFORM)")
    print("=" * 70)

    if os.path.exists(ENRICHED_PARQUET):
        df = pd.read_parquet(ENRICHED_PARQUET)
    elif os.path.exists(MASTER_PARQUET):
        df = pd.read_parquet(MASTER_PARQUET)
    else:
        raise FileNotFoundError("Jalankan 'python run_pipeline.py refresh' terlebih dahulu agar korpus tersedia.")

    if 'sentiment_name' not in df.columns:
        df['sentiment_name'] = df.get('predicted_sentiment', 'Netral')
    if 'emotion' not in df.columns:
        df['emotion'] = df.get('emotion_9', 'Neutral')
    if 'raw_content' not in df.columns:
        df['raw_content'] = df.get('content', df['cleaned_content'])

    print(f"Total corpus tersedia: {len(df):,} baris")
    if len(df) <= target_total:
        sample_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        np.random.seed(42)
        selected_indices = set()

        # Tier 1 (60%): Stratified platform x topic x sentiment
        tier1_target = int(target_total * 0.60)
        platforms = df['platform'].dropna().unique()
        topics = df['topic'].dropna().unique()
        sentiments = df['sentiment_name'].dropna().unique()
        n_groups = max(1, len(platforms) * len(topics) * len(sentiments))
        per_group = max(2, tier1_target // n_groups)

        t1 = []
        for plat in platforms:
            for top in topics:
                for sent in sentiments:
                    grp = df[(df['platform'] == plat) & (df['topic'] == top) & (df['sentiment_name'] == sent)]
                    if len(grp) > 0:
                        t1.extend(grp.sample(min(len(grp), per_group), random_state=42).index.tolist())
        if len(t1) > tier1_target:
            t1 = list(np.random.choice(t1, tier1_target, replace=False))
        selected_indices.update(t1)

        # Tier 2 (20%): Hard cases (negasi, kontras, sarkasme)
        tier2_target = int(target_total * 0.20)
        hard_patterns = [
            r'terima\s*kasih.*(?:gagal|rusak|buruk)', r'hebat.*(?:korupsi|konflik|gaduh)',
            r'tidak\s+bagus', r'tidak\s+berhasil', r'bukan\s+masalah', r'belum\s+maksimal',
            r'\btapi\b', r'\bnamun\b', r'\bsayangnya\b', r'\bpadahal\b', r'\bseharusnya\b',
        ]
        t2 = []
        rem = df.drop(index=list(selected_indices), errors='ignore')
        for pat in hard_patterns:
            m = rem[rem['cleaned_content'].astype(str).str.lower().str.contains(pat, na=False, regex=True)]
            if len(m) > 0:
                for idx in m.sample(min(len(m), 20), random_state=42).index:
                    if idx not in selected_indices and len(t2) < tier2_target:
                        t2.append(idx)
        selected_indices.update(t2)

        # Tier 3: Balance across all 7 platforms including Threads
        all_7_platforms = ['Facebook', 'Instagram', 'News', 'Threads', 'TikTok', 'Twitter', 'YouTube']
        rem3 = df.drop(index=list(selected_indices), errors='ignore')
        t3_needed = target_total - len(selected_indices)
        per_p = max(1, t3_needed // len(all_7_platforms))
        for plat in all_7_platforms:
            p_df = rem3[rem3['platform'] == plat]
            if len(p_df) > 0:
                take = min(len(p_df), per_p)
                selected_indices.update(p_df.sample(take, random_state=42).index.tolist())

        if len(selected_indices) < target_total:
            rem4 = df.drop(index=list(selected_indices), errors='ignore')
            fill_n = min(target_total - len(selected_indices), len(rem4))
            if fill_n > 0:
                selected_indices.update(rem4.sample(fill_n, random_state=42).index.tolist())

        sample_df = df.loc[list(selected_indices)[:target_total]].sample(frac=1.0, random_state=42).reset_index(drop=True)

    cols = [c for c in ['platform', 'topic', 'cleaned_content', 'raw_content', 'sentiment_name', 'emotion', 'author', 'url'] if c in sample_df.columns]
    sample_df[cols].to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Berhasil menyimpan {len(sample_df):,} sampel Gold Standard ke: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
