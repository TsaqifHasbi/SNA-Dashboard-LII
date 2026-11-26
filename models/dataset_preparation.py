"""
Dataset Preparation & Preprocessing Bridge for IndoBERT Sentiment & Emotion Pipeline
Menghubungkan modul models/ dengan tools/cleaning_data.py secara terpadu.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.cleaning_data import cleaning_data, rebuild_master_corpus


def run_consolidation(platform_arg='all', from_backup=False):
    """Menjalankan pembersihan seluruh platform dan membangun master_scraped_corpus."""
    return cleaning_data(platform_arg=platform_arg, from_backup=from_backup)


def consolidate_all_platforms():
    """Mengembalikan DataFrame master_scraped_corpus hasil konsolidasi 7 platform."""
    return cleaning_data(platform_arg='all', from_backup=False)


if __name__ == '__main__':
    run_consolidation()
