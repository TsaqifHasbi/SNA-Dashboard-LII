import os
import re
import glob
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SCRAPE_DIR = os.path.join(BASE_DIR, 'ScrapeData')
PARQUET_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
CSV_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.csv')
MASTER_PARQUET = os.path.join(DATA_DIR, 'master_scraped_corpus.parquet')
MASTER_CSV = os.path.join(DATA_DIR, 'master_scraped_corpus.csv')

# Indonesian Stopwords
STOPWORDS_ID = {
    'dan', 'yang', 'di', 'ini', 'itu', 'ke', 'dari', 'untuk', 'pada', 'dengan',
    'adalah', 'akan', 'atau', 'juga', 'dalam', 'bisa', 'ada', 'tidak', 'sudah',
    'karena', 'oleh', 'saat', 'para', 'seperti', 'lebih', 'harus', 'kami', 'kita',
    'bagi', 'saja', 'masih', 'tentang', 'telah', 'mereka', 'ia', 'itu', 'hal',
    'tersebut', 'kata', 'agar', 'lalu', 'maka', 'namun', 'serta', 'pun', 'yakni',
    'antara', 'setiap', 'tetapi', 'hingga', 'secara', 'begitu', 'bahwa', 'atas',
    'nya', 'se', 'tak', 'yg', 'gak', 'ga', 'gk', 'udah', 'dah', 'lah', 'dong',
    'kok', 'sih', 'kan', 'tuh', 'nih', 'deh', 'ya', 'lo', 'gue', 'lu', 'gw',
    'https', 'http', 'www', 'com', 'rt', 'amp', 'nan', 'none',
}

TOPIC_STOPWORDS = {
    'nu', 'nahdlatul', 'ulama', 'pbnu', 'pwnu', 'pcnu', 'nahdliyin',
    'muhammadiyah', 'pwm', 'pdm', 'persyarikatan',
    'baznas', 'baznaz', 'badan', 'amil', 'nasional',
    'ormas', 'warga', 'lembaga', 'islam', 'indonesia', 'ri',
    'pak', 'bapak', 'bu', 'ibu', 'gus', 'prof', 'kh',
}

ALL_STOPWORDS = STOPWORDS_ID | TOPIC_STOPWORDS

LEGACY_TOPIC_MAP = {
    'Nahdlatul_Ulama': 'Nahdlatul_Ulama',
    'Banom_dan_Warga_NU': 'Nahdlatul_Ulama',
    'Program_dan_Ekonomi_NU': 'Nahdlatul_Ulama',
    'Muhammadiyah': 'Muhammadiyah',
    'BAZNAS': 'BAZNAS',
    'Kelembagaan_BAZNAS': 'BAZNAS',
    'Zakat_dan_Program_BAZNAS': 'BAZNAS',
    'Layanan_dan_Akuntabilitas_BAZNAS': 'BAZNAS',
}


def _normalize_topic_series(df: pd.DataFrame) -> pd.Series:
    raw_t = df['topic'].fillna('').astype(str).str.strip() if 'topic' in df.columns else pd.Series('', index=df.index)
    mapped = raw_t.map(LEGACY_TOPIC_MAP)
    unresolved = mapped.isna()
    if unresolved.any():
        kw_s = df['keyword'].fillna('').astype(str).str.lower() if 'keyword' in df.columns else pd.Series('', index=df.index)
        txt_s = df['cleaned_content'].fillna('').astype(str).str.lower() if 'cleaned_content' in df.columns else pd.Series('', index=df.index)
        comb = kw_s + ' ' + txt_s
        is_baznas = comb.str.contains(r'baznas|baznaz|amil zakat', na=False)
        is_muh = comb.str.contains(r'muhammadiyah|aisyiyah|lazismu|amal usaha', na=False)
        fallback = pd.Series('Nahdlatul_Ulama', index=df.index)
        fallback = fallback.mask(is_muh, 'Muhammadiyah')
        fallback = fallback.mask(is_baznas, 'BAZNAS')
        mapped = mapped.fillna(fallback)
    return mapped


def clean_lii_entities(text: str) -> str:
    """
    Membersihkan kata entitas inti Lembaga Islam Indonesia (NU, Muhammadiyah, BAZNAS)
    dari teks agar WordCloud fokus pada ISU SUBSTANTIF yang dibahas masyarakat.
    """
    if not isinstance(text, str):
        return ""

    patterns = [
        r'\bnahdlatul\s+ulama\b',
        r'\bpengurus\s+besar\s+nahdlatul\s+ulama\b',
        r'\bbadan\s+amil\s+zakat\s+nasional\b',
        r'\bpp\s+muhammadiyah\b',
        r'\bormas\s+muhammadiyah\b',
        r'\bwarga\s+muhammadiyah\b',
        r'\bmuhammadiyah\b',
        r'\bbaznas\s+ri\b',
        r'\blembaga\s+baznas\b',
        r'\bbaznas\b',
        r'\bbaznaz\b',
        r'\bormas\s+nu\b',
        r'\bwarga\s+nu\b',
        r'\bpbnu\b',
        r'\bpwnu\b',
        r'\bpcnu\b',
        r'\bnu\s+online\b',
        r'\bnu\b',
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, ' ', cleaned, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned).strip()


# Alias backward compatibility untuk modul visualisasi
clean_nu_entities = clean_lii_entities
clean_pu_entities = clean_lii_entities


EMOTION_LEXICON = {
    'Joy': ['senang', 'bahagia', 'gembira', 'sukacita', 'ceria', 'riang', 'puas',
            'bangga', 'syukur', 'bersyukur', 'mantap', 'keren', 'hebat', 'luar biasa', 'alhamdulillah', 'berkah'],
    'Trust': ['percaya', 'amanah', 'dukung', 'mendukung', 'apresiasi', 'salut', 'terbaik',
              'istiqomah', 'bermanfaat', 'membantu', 'peduli', 'khidmat', 'solid', 'kompak'],
    'Anticipation': ['semoga', 'berharap', 'harapan', 'rencana', 'target', 'menanti',
                     'persiapan', 'kedepan', 'masa depan', 'program', 'agenda'],
    'Sadness': ['sedih', 'duka', 'pilu', 'haru', 'menyesal', 'kecewa', 'kehilangan',
                'tragis', 'miris', 'prihatin', 'kasihan', 'berduka', 'wafat'],
    'Anger': ['marah', 'kesal', 'geram', 'murka', 'emosi', 'benci', 'muak',
              'kritik', 'protes', 'tolak', 'menolak', 'konflik', 'pecah', 'gagal', 'parah'],
    'Fear': ['takut', 'khawatir', 'cemas', 'was-was', 'ngeri', 'panik',
             'gelisah', 'ragu', 'ancaman', 'bahaya', 'waspada', 'rawan'],
    'Surprise': ['terkejut', 'kaget', 'heran', 'tercengang', 'wow', 'astaga',
                 'waduh', 'ternyata', 'mendadak', 'tiba-tiba', 'heboh', 'viral'],
    'Disgust': ['jijik', 'busuk', 'kotor', 'sampah', 'korupsi', 'bobrok',
                'memalukan', 'najis', 'hina', 'curang', 'culas', 'munafik'],
}


def detect_emotion(text):
    if not isinstance(text, str) or not text.strip():
        return 'Neutral'
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_LEXICON.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[emotion] = score
    if not scores:
        return 'Neutral'
    return max(scores, key=scores.get)


def _baseline_sentiment(text):
    """Fallback sentimen cepat jika korpus masih mentah (belum di-inferensi IndoBERT)."""
    if not isinstance(text, str) or not text.strip():
        return 'Netral'
    t = text.lower()
    pos_kw = ['apresiasi', 'bangga', 'mantap', 'hebat', 'bermanfaat', 'bantuan', 'peduli',
              'alhamdulillah', 'berkah', 'sukses', 'dukung', 'positif', 'baik', 'sejuk', 'damai', 'inspiratif']
    neg_kw = ['kritik', 'kecewa', 'konflik', 'polemik', 'kontroversi', 'tolak', 'gagal',
              'parah', 'korupsi', 'buruk', 'ricuh', 'marah', 'protes', 'masalah', 'pecah', 'bubar']
    p_score = sum(1 for w in pos_kw if w in t)
    n_score = sum(1 for w in neg_kw if w in t)
    if p_score > n_score:
        return 'Positif'
    if n_score > p_score:
        return 'Negatif'
    return 'Netral'


def extract_emojis(text):
    if not isinstance(text, str):
        return []
    # pyrefly: ignore[missing-import]
    import emoji
    return [c for c in text if c in emoji.EMOJI_DATA]


def _empty_schema_df():
    now_ts = pd.Timestamp.now()
    return pd.DataFrame({
        'platform': pd.Series(dtype='str'),
        'topic': pd.Series(dtype='str'),
        'keyword': pd.Series(dtype='str'),
        'author': pd.Series(dtype='str'),
        'handle': pd.Series(dtype='str'),
        'raw_content': pd.Series(dtype='str'),
        'cleaned_content': pd.Series(dtype='str'),
        'url': pd.Series(dtype='str'),
        'created_at': pd.Series(dtype='str'),
        'date': pd.Series(dtype='datetime64[ns]'),
        'date_str': pd.Series(dtype='str'),
        'year_month': pd.Series(dtype='str'),
        'likes': pd.Series(dtype='int32'),
        'comments': pd.Series(dtype='int32'),
        'shares': pd.Series(dtype='int32'),
        'reposts': pd.Series(dtype='int32'),
        'views': pd.Series(dtype='int32'),
        'total_engagement': pd.Series(dtype='int32'),
        'reach': pd.Series(dtype='int32'),
        'sentiment_label': pd.Series(dtype='int8'),
        'sentiment_name': pd.Series(dtype='str'),
        'emotion': pd.Series(dtype='str'),
    })


def _load_fallback_scrape_csvs():
    """Fallback otomatis memuat data CSV hasil scraping di ScrapeData/ jika korpus data/ belum dibuat."""
    csv_files = sorted(glob.glob(os.path.join(SCRAPE_DIR, 'scraped_*_Only.csv')))
    frames = []
    for path in csv_files:
        try:
            part = pd.read_csv(path, low_memory=False)
            if len(part) > 0:
                frames.append(part)
        except Exception:
            continue
    if not frames:
        return _empty_schema_df()
    df = pd.concat(frames, ignore_index=True)
    if 'content' in df.columns and 'cleaned_content' not in df.columns:
        df['cleaned_content'] = df['content'].fillna('').astype(str)
    if 'content' in df.columns and 'raw_content' not in df.columns:
        df['raw_content'] = df['content'].fillna('').astype(str)
    if 'retweets' in df.columns and 'reposts' not in df.columns:
        df['reposts'] = df['retweets']
    return df


@st.cache_resource(show_spinner=False)
def _load_main_data_cached(cache_key: str):
    target_file = None
    for p in [PARQUET_PATH, CSV_PATH, MASTER_PARQUET, MASTER_CSV]:
        if os.path.exists(p):
            target_file = p
            break

    if target_file:
        if target_file.endswith('.parquet'):
            try:
                df = pd.read_parquet(target_file, engine='pyarrow')
            except Exception:
                df = pd.read_parquet(target_file)
        else:
            df = pd.read_csv(target_file, low_memory=False)
    else:
        df = _load_fallback_scrape_csvs()

    if len(df) == 0:
        return _empty_schema_df()

    unused_cols = ['thumbnail', 'duration', 'saves', 'views_text', 'description', 'scraped_at', 'post_type', 'source']
    df.drop(columns=[c for c in unused_cols if c in df.columns], inplace=True, errors='ignore')

    if 'content' in df.columns and 'cleaned_content' not in df.columns:
        df['cleaned_content'] = df['content']
    if 'cleaned_content' not in df.columns:
        df['cleaned_content'] = df.get('raw_content', '')
    if 'raw_content' not in df.columns:
        df['raw_content'] = df['cleaned_content']

    for col in ['cleaned_content', 'raw_content', 'handle', 'url', 'keyword']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
        else:
            df[col] = ''

    if 'author' in df.columns:
        df['author'] = df['author'].fillna(df['handle']).replace('', 'anonim').astype(str)
    else:
        df['author'] = 'anonim'

    if 'url' in df.columns:
        norm_url = df['url'].str.replace('/photo/', '/video/', regex=False)
        valid_url = norm_url.str.len() > 10
        dup_mask = valid_url & norm_url.duplicated(keep='first')
        if dup_mask.any():
            df.drop(index=df.index[dup_mask], inplace=True)
        del norm_url, valid_url, dup_mask

    if 'platform' in df.columns:
        df['platform'] = df['platform'].fillna('Other').astype(str)
    else:
        df['platform'] = 'Other'

    df['topic'] = _normalize_topic_series(df)

    for col in ['likes', 'comments', 'shares', 'reposts', 'views', 'total_engagement', 'reach']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0, upper=500_000).astype('int32')
        else:
            df[col] = 0

    df['total_engagement'] = (df['likes'] + df['comments'] + df['shares'] + df['reposts']).astype('int32')
    df['reach'] = (df['views'] + (df['likes'] * 3) + (df['comments'] * 5) + (df['shares'] * 10) + (df['reposts'] * 10)).astype('int32')

    if 'sentiment_name' not in df.columns:
        if 'sentiment_label' in df.columns:
            df['sentiment_label'] = pd.to_numeric(df['sentiment_label'], errors='coerce').fillna(1).astype('int8')
            df['sentiment_name'] = df['sentiment_label'].map({0: 'Negatif', 1: 'Netral', 2: 'Positif'}).fillna('Netral')
        else:
            df['sentiment_name'] = df['cleaned_content'].apply(_baseline_sentiment)
            df['sentiment_label'] = df['sentiment_name'].map({'Negatif': 0, 'Netral': 1, 'Positif': 2}).fillna(1).astype('int8')
    elif 'sentiment_label' not in df.columns:
        df['sentiment_label'] = df['sentiment_name'].map({'Negatif': 0, 'Netral': 1, 'Positif': 2}).fillna(1).astype('int8')
    else:
        df['sentiment_label'] = pd.to_numeric(df['sentiment_label'], errors='coerce').fillna(1).astype('int8')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    if df['date'].isna().all() and 'created_at' in df.columns:
        df['date'] = pd.to_datetime(df['created_at'].astype(str).str.extract(r'(\d{4}-\d{2}-\d{2})')[0], errors='coerce')

    if df['date'].isna().any():
        valid_median = df['date'].dropna().median()
        fallback_dt = valid_median if pd.notna(valid_median) else pd.Timestamp.now().normalize()
        df['date'] = df['date'].fillna(fallback_dt)

    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    if 'emotion' not in df.columns:
        df['emotion'] = df['cleaned_content'].apply(detect_emotion)

    if 'emotion' in df.columns and 'sentiment_name' in df.columns:
        mismatched = (df['sentiment_name'] == 'Negatif') & (df['emotion'].isin(['Joy', 'Trust', 'Anticipation']))
        if mismatched.any():
            try:
                from src.sentiment_corrector import align_emotion_valence_9class
                df.loc[mismatched, 'emotion'] = [
                    align_emotion_valence_9class(t, s, e)
                    for t, s, e in zip(df.loc[mismatched, 'cleaned_content'], df.loc[mismatched, 'sentiment_name'], df.loc[mismatched, 'emotion'])
                ]
            except Exception:
                pass

    cat_cols = ['platform', 'topic', 'sentiment_name', 'emotion', 'predicted_sentiment', 'emotion_sentiment']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype('category')

    import gc
    gc.collect()
    return df


def load_main_data():
    for p in [PARQUET_PATH, CSV_PATH, MASTER_PARQUET, MASTER_CSV]:
        if os.path.exists(p):
            return _load_main_data_cached(f"{p}:{os.path.getmtime(p)}")
    csv_files = sorted(glob.glob(os.path.join(SCRAPE_DIR, 'scraped_*_Only.csv')))
    sig = "|".join(f"{os.path.basename(f)}:{os.path.getmtime(f)}" for f in csv_files)
    return _load_main_data_cached(f"scrape:{sig}")


def filter_dataset(df, selected_topic='Semua', selected_platform='Semua', start_date=None, end_date=None):
    if df is None or len(df) == 0:
        return df
    mask = None

    if selected_topic and selected_topic != 'Semua':
        m = (df['topic'] == selected_topic)
        mask = m if mask is None else (mask & m)

    if selected_platform and selected_platform != 'Semua':
        m = (df['platform'] == selected_platform)
        mask = m if mask is None else (mask & m)

    if start_date and end_date and 'date' in df.columns:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        m = (df['date'] >= start_ts) & (df['date'] <= end_ts)
        mask = m if mask is None else (mask & m)

    if mask is None:
        return df
    return df[mask]
