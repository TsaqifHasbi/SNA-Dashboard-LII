"""
Pipeline Konsolidasi, Smart NLP Cleaning, dan Pemisahan Kandidat Hapus per Platform (7 Platform)
=================================================================================================
Proyek: LEMBAGA ISLAM INDONESIA (LII) — 3 Topik Utama:
  1. Nahdlatul_Ulama (NU)
  2. Muhammadiyah
  3. BAZNAS
"""

import os
import sys
import re
import html
import glob
import shutil
import argparse
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
COLAB_DIR = os.path.join(PROJECT_ROOT, 'colab_cells')
if COLAB_DIR not in sys.path:
    sys.path.insert(0, COLAB_DIR)

# pyrefly: ignore [missing-import]
from cell_02_config import (
    PROJECT_TITLE,
    DEFAULT_TOPIC,
    normalize_canonical_topic,
    get_date_range_label,
    get_scrape_since,
    get_scrape_until,
    is_in_ta_date_range,
    parse_to_standard_wib,
    atomic_save_csv_df,
)
from tools.relevance_filter import classify_lii_relevance

SCRAPE_DIR = os.path.join(PROJECT_ROOT, 'ScrapeData')
BACKUP_DIR = os.path.join(SCRAPE_DIR, '_backup_pre_cleaning')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

CSV_FILES = {
    'Facebook':  'scraped_Facebook_Only.csv',
    'Instagram': 'scraped_Instagram_Only.csv',
    'News':      'scraped_News_Only.csv',
    'Threads':   'scraped_Threads_Only.csv',
    'TikTok':    'scraped_TikTok_Only.csv',
    'Twitter':   'scraped_Twitter_Only.csv',
    'YouTube':   'scraped_YouTube_Only.csv',
}

PLATFORM_ALIASES = {
    'all': 'ALL',
    'facebook': 'Facebook', 'fb': 'Facebook',
    'instagram': 'Instagram', 'ig': 'Instagram',
    'news': 'News', 'berita': 'News',
    'threads': 'Threads', 'th': 'Threads',
    'tiktok': 'TikTok', 'tt': 'TikTok',
    'twitter': 'Twitter', 'x': 'Twitter',
    'youtube': 'YouTube', 'yt': 'YouTube',
}

URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_CLEAN_RE = re.compile(r'@[\w\.-]+[:,\s]?')
WS_MULTI_RE = re.compile(r'\s{2,}')
LINE_BREAK_RE = re.compile(r'[\r\n\t]+')
HASHTAG_EXTRACT_RE = re.compile(r'#(\w+)')
MENTION_EXTRACT_RE = re.compile(r'@([\w\.-]+)')

try:
    # pyrefly: ignore [missing-import]
    import emoji
    HAS_EMOJI_LIB = True
except ImportError:
    HAS_EMOJI_LIB = False


def extract_emojis(text: str) -> str:
    if not text or not HAS_EMOJI_LIB or text.isascii():
        return ""
    emojis = [c for c in text if c in emoji.EMOJI_DATA]
    return " ".join(emojis) if emojis else ""


def clean_text_for_bert_keep_hashtag(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = html.unescape(text)
    t = URL_RE.sub('', t)
    t = MENTION_CLEAN_RE.sub(' ', t)
    t = LINE_BREAK_RE.sub(' ', t)
    t = WS_MULTI_RE.sub(' ', t)
    t = "".join(ch for ch in t if ch.isprintable())
    return t.strip()


def _safe_int(val, max_val=50_000_000):
    try:
        if pd.isna(val):
            return 0
        v = int(float(val))
        return max(0, min(v, max_val))
    except Exception:
        return 0


def clean_legacy_backup_files():
    for pat in [os.path.join(SCRAPE_DIR, "*_BEFORE_NORM_*.csv")]:
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except Exception:
                pass


def _merge_preserve_kandidat(kandidat_csv_path: str, new_rows: list, kept_urls: set) -> pd.DataFrame:
    """Menggabungkan arsip Kandidat_Hapus lama dengan eliminasi baru tanpa pernah menghapus arsip lama."""
    frames = []
    if os.path.exists(kandidat_csv_path) and os.path.getsize(kandidat_csv_path) > 0:
        try:
            df_prev = pd.read_csv(kandidat_csv_path, low_memory=False)
            if not df_prev.empty:
                frames.append(df_prev)
        except Exception:
            pass
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if not df_new.empty:
            frames.append(df_new)
    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    if 'url' in df_all.columns and kept_urls:
        url_str = df_all['url'].fillna('').astype(str).str.strip()
        df_all = df_all[~url_str.isin(kept_urls)].copy()

    if 'url' in df_all.columns and not df_all.empty:
        url_str = df_all['url'].fillna('').astype(str).str.strip()
        has_u = url_str != ''
        df_u = df_all[has_u].drop_duplicates(subset=['url'], keep='last')
        df_no = df_all[~has_u]
        if not df_no.empty and 'content' in df_no.columns:
            dedup_cols = [c for c in ['content', 'alasan_eliminasi'] if c in df_no.columns]
            if dedup_cols:
                df_no = df_no.drop_duplicates(subset=dedup_cols, keep='last')
        df_all = pd.concat([df_u, df_no], ignore_index=True)
    return df_all


def clean_single_platform(platform: str, fname: str, from_backup: bool = False, dry_run: bool = False):
    backup_fpath = os.path.join(BACKUP_DIR, fname)
    live_fpath = os.path.join(SCRAPE_DIR, fname)
    kandidat_fpath = os.path.join(SCRAPE_DIR, f"scraped_{platform}_Kandidat_Hapus.csv")

    if from_backup and os.path.exists(backup_fpath):
        input_fpath = backup_fpath
        source_label = f"Backup ({os.path.basename(backup_fpath)})"
    elif os.path.exists(live_fpath):
        input_fpath = live_fpath
        source_label = f"Live ScrapeData ({os.path.basename(live_fpath)})"
        if not dry_run:
            os.makedirs(BACKUP_DIR, exist_ok=True)
            snapshot_path = os.path.join(BACKUP_DIR, f"{fname.replace('.csv', '')}_snapshot_raw.csv")
            if not os.path.exists(snapshot_path):
                try:
                    shutil.copy2(live_fpath, snapshot_path)
                except Exception:
                    pass
    elif os.path.exists(backup_fpath):
        input_fpath = backup_fpath
        source_label = f"Backup ({os.path.basename(backup_fpath)})"
    else:
        print(f"[SKIP] File {fname} belum tersedia di ScrapeData/. Dilewati.")
        return None

    print(f"\n[{platform.upper()}] Memproses dari: {source_label}...")
    try:
        df = pd.read_csv(input_fpath, low_memory=False)
    except Exception as e:
        print(f"[ERROR] Gagal membaca CSV {input_fpath}: {e}")
        return None

    total_raw = len(df)
    accepted_rows = []
    kandidat_hapus_rows = []
    drop_counts = Counter()
    reconciled_count = 0

    seen_urls_in_platform = set()
    seen_texts_without_url = set()
    since_lbl = get_scrape_since()
    until_lbl = get_scrape_until()

    def _reject(row_obj, reason_code, reason_detail):
        r_del = dict(row_obj)
        r_del['alasan_eliminasi'] = reason_code
        r_del['keterangan'] = reason_detail
        kandidat_hapus_rows.append(r_del)
        drop_counts[reason_code] += 1

    for row in df.to_dict('records'):
        created_at_raw = str(row.get('created_at', '') or '').strip()
        date_raw = str(row.get('date', '') or '').strip()
        target_date_eval = created_at_raw if created_at_raw.lower() not in ['nan', 'none', 'tidak_ditemukan', ''] else date_raw

        if not target_date_eval or target_date_eval.lower() in ['nan', 'none', 'tidak_ditemukan', '']:
            _reject(row, 'TANGGAL_TIDAK_DITEMUKAN', 'Tanggal postingan kosong atau tidak tersedia')
            continue

        if not is_in_ta_date_range(target_date_eval):
            _reject(row, 'DI_LUAR_RENTANG_WAKTU', f'Tanggal ({target_date_eval}) di luar rentang aktif {since_lbl} s.d. {until_lbl}')
            continue

        wib_str, date_str, y = parse_to_standard_wib(target_date_eval)
        if not wib_str or not date_str:
            _reject(row, 'FORMAT_TANGGAL_TIDAK_VALID', f'Format tanggal tidak dapat diparsing: {target_date_eval}')
            continue

        content = str(row.get('content', '') or '').strip()
        desc = str(row.get('description', '') or '').strip()
        if content.lower() == 'nan':
            content = ''
        if desc.lower() == 'nan':
            desc = ''

        if platform in ('YouTube', 'News'):
            raw_content = f"{content}. {desc}".strip('. ') if (desc and desc != content) else content
        else:
            raw_content = content if content else desc

        if not raw_content:
            _reject(row, 'TEKS_KOSONG', 'Tidak memiliki konten teks judul atau deskripsi')
            continue

        cleaned_content = clean_text_for_bert_keep_hashtag(raw_content)
        if not cleaned_content or not cleaned_content.split():
            _reject(row, 'TEKS_KOSONG', 'Konten hanya berisi URL/karakter tidak terbaca')
            continue

        url_raw = row.get('url', '')
        url_val = str(url_raw).strip() if pd.notna(url_raw) and str(url_raw).strip().lower() != 'nan' else ''
        if url_val:
            if url_val in seen_urls_in_platform:
                _reject(row, 'DUPLIKAT_URL', f'Tautan URL duplikat: {url_val}')
                continue
            seen_urls_in_platform.add(url_val)
        else:
            norm_key = WS_MULTI_RE.sub(' ', cleaned_content.lower()).strip()
            if norm_key in seen_texts_without_url:
                _reject(row, 'DUPLIKAT_TEKS_TANPA_URL', 'Teks identik tanpa URL pembeda')
                continue
            seen_texts_without_url.add(norm_key)

        kw_val = str(row.get('keyword', '') or '').strip()
        if kw_val.lower() == 'nan':
            kw_val = ''

        row_topic = str(row.get('topic', '') or '').strip()
        prelim_topic = normalize_canonical_topic(row_topic, content=raw_content, keyword=kw_val)

        # Pemilahan relevansi konteks multi-lapis (bahasa asing, Sunda 'nu', homonim, entitas LII)
        is_rel, rel_code, rel_detail, resolved_topic = classify_lii_relevance(
            raw_content,
            topic_hint=prelim_topic,
            keyword=kw_val,
            platform=platform,
        )
        if not is_rel:
            _reject(row, rel_code, rel_detail)
            continue
        topic_val = resolved_topic or prelim_topic
        if topic_val != prelim_topic:
            reconciled_count += 1

        hashtags_list = HASHTAG_EXTRACT_RE.findall(raw_content)
        hashtags_str = ' '.join(['#' + h for h in hashtags_list]) if hashtags_list else ''

        mentions_list = MENTION_EXTRACT_RE.findall(raw_content)
        mentions_str = ','.join(mentions_list) if mentions_list else ''
        emojis_str = extract_emojis(raw_content)

        likes = _safe_int(row.get('likes', 0))
        comments = _safe_int(row.get('comments', 0))
        shares = _safe_int(row.get('shares', 0))
        reposts = _safe_int(row.get('reposts', row.get('retweets', 0)))
        views = _safe_int(row.get('views', 0))
        total_engagement = likes + comments + shares + reposts
        reach = views + (likes * 3) + (comments * 5) + (shares * 10) + (reposts * 10)

        author = str(row.get('author', '') or '').strip()
        handle = str(row.get('handle', '') or '').strip()
        if author.lower() == 'nan':
            author = ''
        if handle.lower() == 'nan':
            handle = ''

        user_id = handle or author or f"{platform}_User"

        post_id_val = str(row.get('post_id', '') or '').strip()
        if post_id_val.lower() == 'nan' or not post_id_val:
            post_id_val = f"{platform.lower()}_{abs(hash(url_val or cleaned_content)) % 10_000_000:07d}"

        clean_item = {
            'post_id': post_id_val,
            'platform': platform,
            'topic': topic_val,
            'keyword': kw_val,
            'user_id': user_id,
            'author': author or user_id,
            'handle': handle or user_id,
            'url': url_val,
            'content': raw_content,
            'raw_content': raw_content,
            'cleaned_content': cleaned_content,
            'hashtags': hashtags_str,
            'mentions': mentions_str,
            'emojis': emojis_str,
            'created_at': wib_str,
            'date': date_str,
            'year': y,
            'year_month': date_str[:7] if len(date_str) >= 7 else '',
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'reposts': reposts,
            'views': views,
            'total_engagement': total_engagement,
            'reach': reach,
        }
        if pd.notna(row.get('source')) and str(row.get('source')).strip().lower() != 'nan':
            clean_item['source'] = str(row.get('source')).strip()
        if pd.notna(row.get('post_type')) and str(row.get('post_type')).strip().lower() != 'nan':
            clean_item['post_type'] = str(row.get('post_type')).strip()
        accepted_rows.append(clean_item)

    clean_csv_path = os.path.join(SCRAPE_DIR, f"scraped_{platform}_Only.csv")
    df_clean = pd.DataFrame(accepted_rows)
    if not df_clean.empty and 'cleaned_content' in df_clean.columns:
        norm_series = df_clean['cleaned_content'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        freq_map = norm_series.value_counts().to_dict()
        df_clean['content_frequency'] = norm_series.map(freq_map).fillna(1).astype(int)
        df_clean['is_repeated_content'] = (df_clean['content_frequency'] > 1).astype(int)

    kandidat_csv_path = os.path.join(SCRAPE_DIR, f"scraped_{platform}_Kandidat_Hapus.csv")
    kept_url_set = set(df_clean['url'].dropna().astype(str).str.strip()) if (not df_clean.empty and 'url' in df_clean.columns) else set()
    df_kandidat = _merge_preserve_kandidat(kandidat_csv_path, kandidat_hapus_rows, kept_url_set)

    if not dry_run:
        atomic_save_csv_df(df_clean, clean_csv_path, encoding='utf-8-sig')
        if not df_kandidat.empty or not os.path.exists(kandidat_csv_path):
            atomic_save_csv_df(df_kandidat, kandidat_csv_path, encoding='utf-8-sig')

    n_acc = len(accepted_rows)
    n_del_new = len(kandidat_hapus_rows)
    n_del_total = len(df_kandidat)
    n_buzzer = int((df_clean['is_repeated_content'] == 1).sum()) if not df_clean.empty and 'is_repeated_content' in df_clean.columns else 0
    pct = (n_acc / total_raw) * 100 if total_raw > 0 else 0.0
    print(f"   • Total Data Diproses   : {total_raw:>7,} baris")
    print(f"   • Data Bersih Lolos     : {n_acc:>7,} baris ({pct:5.1f}%) -> {os.path.basename(clean_csv_path)}")
    print(f"   • Kandidat Hapus        : +{n_del_new:,} baru | Total arsip terjaga: {n_del_total:>7,} baris -> {os.path.basename(kandidat_csv_path)}")
    if drop_counts:
        breakdown = ", ".join(f"{k}={v:,}" for k, v in drop_counts.most_common())
        print(f"   • Rincian Eliminasi     : {breakdown}")
    if reconciled_count:
        print(f"   • Rekonsiliasi Topik    : {reconciled_count:,} baris diselaraskan ke topik entitas sebenarnya")

    out_context_cnt = sum(v for k, v in drop_counts.items() if k.startswith('OUT_OF_CONTEXT_'))
    return {
        'Platform': platform,
        'Total Input': total_raw,
        'Bersih (Only)': n_acc,
        'Out-of-Context': out_context_cnt,
        'Buzzer/Berulang': n_buzzer,
        'Kandidat Hapus': n_del_total,
        'Persen Lolos': f"{pct:.1f}%",
    }


def rebuild_master_corpus():
    all_dfs = []
    print("\n" + "=" * 85)
    print(f"📦 MENGGABUNGKAN SELURUH DATA BERSIH KE MASTER CORPUS ({PROJECT_TITLE})...")
    print("=" * 85)

    for p in CSV_FILES.keys():
        c_path = os.path.join(SCRAPE_DIR, f"scraped_{p}_Only.csv")
        if os.path.exists(c_path):
            try:
                sub_df = pd.read_csv(c_path, low_memory=False)
                if len(sub_df) > 0:
                    if 'platform' not in sub_df.columns:
                        sub_df['platform'] = p
                    all_dfs.append(sub_df)
                    print(f"   ✓ {p:<12}: {len(sub_df):>7,} baris bersih termuat")
            except Exception as e:
                print(f"   ✗ [WARN] Gagal membaca {c_path}: {e}")

    if not all_dfs:
        print("[WARN] Tidak ada data bersih yang ditemukan di ScrapeData/.")
        return pd.DataFrame()

    df_master = pd.concat(all_dfs, ignore_index=True)

    if 'cleaned_content' not in df_master.columns:
        df_master['cleaned_content'] = df_master['content'].fillna('').astype(str).map(clean_text_for_bert_keep_hashtag)

    norm_series = df_master['cleaned_content'].fillna('').astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    freq_map = norm_series.value_counts().to_dict()
    df_master['content_frequency_global'] = norm_series.map(freq_map).fillna(1).astype(int)
    df_master['is_cross_platform_buzzer'] = (df_master['content_frequency_global'] > 1).astype(int)

    master_csv = os.path.join(DATA_DIR, 'master_scraped_corpus.csv')
    master_parquet = os.path.join(DATA_DIR, 'master_scraped_corpus.parquet')

    df_master.to_csv(master_csv, index=False, encoding='utf-8-sig')
    df_master.to_parquet(master_parquet, index=False)

    print("-" * 85)
    print(f"🎉 MASTER CORPUS LII BERHASIL DIPERBARUI: {len(df_master):,} baris ({len(all_dfs)} platform)")
    print("   Distribusi Topik:", df_master['topic'].value_counts().to_dict())
    print(f"   - CSV    : {master_csv} ({os.path.getsize(master_csv) / (1024*1024):.2f} MB)")
    print(f"   - Parquet: {master_parquet} ({os.path.getsize(master_parquet) / (1024*1024):.2f} MB)")
    return df_master


def cleaning_data(platform_arg=None, from_backup=False, dry_run=False):
    if platform_arg is None:
        parser = argparse.ArgumentParser(description=f"Pipeline Cleaning & Konsolidasi Data {PROJECT_TITLE}")
        parser.add_argument('platform_pos', nargs='?', default=None)
        parser.add_argument('--platform', '-p', type=str, default=None)
        parser.add_argument('--from-backup', action='store_true')
        parser.add_argument('--dry-run', action='store_true', help='Simulasi cleaning tanpa menimpa file CSV')
        args, _ = parser.parse_known_args()
        raw_p = args.platform or args.platform_pos or 'all'
        from_backup = args.from_backup
        dry_run = args.dry_run
    else:
        raw_p = platform_arg

    target_platform = PLATFORM_ALIASES.get(str(raw_p).strip().lower(), raw_p)
    print("=" * 85)
    mode_lbl = " [DRY-RUN SIMULASI]" if dry_run else ""
    print(f"🚀 PIPELINE KONSOLIDASI & CONTEXT-AWARE CLEANING — {PROJECT_TITLE}{mode_lbl}")
    print(f"📅 Rentang Waktu Aktif: {get_date_range_label()}")
    print("=" * 85)

    if not dry_run:
        clean_legacy_backup_files()
    summary_table = []

    if str(target_platform).upper() == 'ALL':
        for platform, fname in CSV_FILES.items():
            res = clean_single_platform(platform, fname, from_backup=from_backup, dry_run=dry_run)
            if res:
                summary_table.append(res)
    else:
        matched_platform = next((p for p in CSV_FILES if p.lower() == str(target_platform).lower()), None)
        if not matched_platform:
            print(f"[ERROR] Platform '{target_platform}' tidak dikenali!")
            return pd.DataFrame()
        res = clean_single_platform(matched_platform, CSV_FILES[matched_platform], from_backup=from_backup, dry_run=dry_run)
        if res:
            summary_table.append(res)

    df_master = pd.DataFrame() if dry_run else rebuild_master_corpus()
    if summary_table:
        print("\n" + "=" * 85)
        print(pd.DataFrame(summary_table).to_string(index=False))
        print("=" * 85)
    return df_master


def run_consolidation(platform_arg='all', from_backup=False):
    return cleaning_data(platform_arg=platform_arg, from_backup=from_backup)


def consolidate_all_platforms():
    return cleaning_data(platform_arg='all', from_backup=False)


if __name__ == '__main__':
    cleaning_data()
