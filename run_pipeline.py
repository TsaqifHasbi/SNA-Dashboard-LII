"""
Unified End-to-End Pipeline Orchestrator — LEMBAGA ISLAM INDONESIA (LII)
3 Topik Utama: Nahdlatul_Ulama (NU), Muhammadiyah, & BAZNAS
========================================================================
Contoh penggunaan dari root proyek:
  1. Scraping khusus topik baru (misal Muhammadiyah saja):
       python run_pipeline.py scrape --platform all --topic Muhammadiyah
       python run_pipeline.py scrape --platform youtube --topic Muhammadiyah
  2. Sinkronisasi cepat korpus & SNA setelah scraping selesai:
       python run_pipeline.py refresh
  3. Pipeline penuh termasuk Fine-Tuning IndoBERT (3 Sentimen + 9 Emosi):
       python run_pipeline.py all
"""

import os
import sys
import time
import argparse
import subprocess

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
COLAB_DIR = os.path.join(PROJECT_ROOT, 'colab_cells')
if COLAB_DIR not in sys.path:
    sys.path.insert(0, COLAB_DIR)

from cell_02_config import PROJECT_TITLE, get_date_range_label

SCRAPER_SCRIPTS = {
    'youtube': 'run_youtube_only.py', 'yt': 'run_youtube_only.py',
    'news': 'run_news_only.py', 'berita': 'run_news_only.py',
    'twitter': 'run_twitter_only.py', 'x': 'run_twitter_only.py',
    'tiktok': 'run_tiktok_only.py', 'tt': 'run_tiktok_only.py',
    'instagram': 'run_instagram_only.py', 'ig': 'run_instagram_only.py',
    'threads': 'run_threads_only.py', 'th': 'run_threads_only.py',
    'facebook': 'run_facebook_only.py', 'fb': 'run_facebook_only.py',
}


def step_scrape(platform='all', topic=None, keyword=None):
    plat = platform.strip().lower()
    if plat == 'all':
        order = ['news', 'youtube', 'twitter', 'tiktok', 'threads', 'instagram', 'facebook']
    else:
        if plat not in SCRAPER_SCRIPTS:
            raise ValueError(f"Platform '{platform}' tidak dikenali. Pilihan: {', '.join(sorted(set(SCRAPER_SCRIPTS.keys())))} atau 'all'")
        order = [plat]

    for p in order:
        script_path = os.path.join(COLAB_DIR, SCRAPER_SCRIPTS[p])
        cmd = [sys.executable, '-u', script_path]
        if topic and topic.lower() != 'all':
            cmd.extend(['--topic', topic])
        if keyword:
            cmd.extend(['--keyword', keyword])
        print(f"\n🚀 Menjalankan Scraper [{p.upper()}]: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)


def step_clean(platform='all', dry_run=False):
    from tools.cleaning_data import cleaning_data
    return cleaning_data(platform_arg=platform, dry_run=dry_run)


def step_prepare():
    from models.label_training_data import prepare_labeled_dataset
    from models.prepare_9_emotions_dataset import load_and_standardize_corpus, generate_balanced_splits
    prepare_labeled_dataset()
    df_c = load_and_standardize_corpus()
    generate_balanced_splits(df_c)


def step_train(epochs_sent=3, epochs_emo=4, batch_size=16):
    from models.train_indobert import run_training_pipeline as train_sent
    from models.train_indobert_9_emotions import run_training_pipeline as train_emo
    train_sent(epochs=epochs_sent, batch_size=batch_size)
    train_emo(epochs=epochs_emo, batch_size=batch_size)


def step_infer(batch_size=64):
    from models.infer_sentiment import batch_infer_sentiment
    return batch_infer_sentiment(batch_size=batch_size)


def step_sna():
    from sna.build_sentiment_sna import build_sna_by_sentiment
    return build_sna_by_sentiment()


def main():
    parser = argparse.ArgumentParser(description=f"End-to-End Orchestrator — {PROJECT_TITLE}")
    parser.add_argument(
        'stage',
        choices=['scrape', 'clean', 'prepare', 'gold-sample', 'gold-annotate', 'train', 'infer', 'sna', 'refresh', 'all'],
        help="Tahap pipeline yang ingin dijalankan"
    )
    parser.add_argument('--platform', '-p', default='all', help="Filter platform untuk tahap scrape / clean")
    parser.add_argument('--topic', '-t', default=None, help="Filter topik untuk scrape: Nahdlatul_Ulama, Muhammadiyah, atau BAZNAS")
    parser.add_argument('--keyword', '-k', default=None, help="Filter satu keyword spesifik saat scrape")
    parser.add_argument('--batch-size', type=int, default=64, help="Ukuran batch untuk inferensi / training")
    parser.add_argument('--dry-run', action='store_true', help="Khusus tahap clean: simulasi filter tanpa menimpa CSV")
    parser.add_argument('--skip-clean', action='store_true', help="Khusus tahap refresh: langsung infer + sna hanya dari data *_Only.csv / master_scraped_corpus tanpa menyentuh cleaning/Kandidat_Hapus")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 80)
    print(f"⚡ ORCHESTRATOR PIPELINE TERPADU — {PROJECT_TITLE}")
    print(f"📅 Rentang Tanggal Aktif: {get_date_range_label()}")
    print(f"🎯 Mode Eksekusi        : {args.stage.upper()}")
    print("=" * 80)

    if args.stage == 'scrape':
        step_scrape(args.platform, topic=args.topic, keyword=args.keyword)
    elif args.stage == 'clean':
        step_clean(args.platform, dry_run=args.dry_run)
    elif args.stage == 'prepare':
        step_prepare()
    elif args.stage == 'gold-sample':
        from tools.step1_sample_gold_standard import main as run_gs1
        run_gs1()
    elif args.stage == 'gold-annotate':
        from tools.step2_gemini_annotation import main as run_gs2
        run_gs2()
    elif args.stage == 'train':
        step_train(batch_size=min(args.batch_size, 32))
    elif args.stage == 'infer':
        step_infer(batch_size=args.batch_size)
    elif args.stage == 'sna':
        step_sna()
    elif args.stage == 'refresh':
        if not args.skip_clean:
            step_clean(args.platform)
        else:
            print("⏭️  [--skip-clean] Melewati cleaning — hanya fokus pada master_scraped_corpus (data *_Only.csv) -> infer -> sna.")
        step_infer(batch_size=args.batch_size)
        step_sna()
    elif args.stage == 'all':
        step_clean(args.platform)
        step_prepare()
        step_train(batch_size=min(args.batch_size, 32))
        step_infer(batch_size=args.batch_size)
        step_sna()

    print(f"\n✨ Tahap '{args.stage}' tuntas dalam {time.time() - t0:.1f} detik!")


if __name__ == '__main__':
    main()
