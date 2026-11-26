import os
import sys
import io
import shutil
import pandas as pd
import re
import html
import unicodedata

# Fix console encoding on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPE_DIR = os.path.join(BASE_DIR, 'ScrapeData')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_PATH = os.path.join(BASE_DIR, 'colab_cells', 'cell_02_config.py')
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = r'd:\Project_Streamlit\cobaAPI\colab_cells\cell_02_config.py'

# 1. Konfigurasi Topik dan Path
COBA_SCRAPE_DIR = r'd:\Project_Streamlit\cobaAPI\ScrapeData'
PLATFORMS_TO_FILTER = ['Twitter', 'Instagram', 'TikTok', 'YouTube', 'Facebook', 'News']


# 2. Pola Regex Relevansi Inti (Core + Hashtags Lengkap Termasuk Prefix/Suffix Tanpa Spasi)
pat_mbg_core = re.compile(
    r'\bmbg\w*|'
    r'#\w*mbg\w*|'  # Menangkap #stopmbg, #STOPMBG, #keracunanmbg, #korupsimbg, #tolakmbg, #hapusmbg, #menumbg, #menumbghariini, dll
    r'#?\w*makan(?:an)?[\s-]+bergizi[\s-]+gratis\w*|'
    r'#?\w*makan(?:an)?[\s-]+gratis[\s-]+bergizi\w*|'
    r'#?\w*makan[\s-]+siang[\s-]+gratis\w*|'
    r'#?\w*makan[\s-]+gratis[\s-]+siang\w*|'
    r'#?\w*makan[\s-]+(?:tidak|tak|bukan|gak)?[\s-]*bergizi[\s-]+(?:tidak|tak|bukan|gak)?[\s-]*gratis\w*|'
    r'makan\s*(?:\([^\)]+\)|tidak|tak|bukan|gak)\s*bergizi|'
    r'makan(?:an)?\s*bergizi\s*(?:\([^\)]+\)|tidak|tak|bukan|gak)\s*gratis|'
    r'#?\w*makan(?:an|ang)?[\s-]+gratis[\s-]+prabowo\w*|'
    r'#?\w*makan(?:an|ang)?[\s-]+gratis[\s-]+gibran\w*|'
    r'#?\w*makangratisprabowo\w*|'
    r'#?\w*makanangratisprabowo\w*|'
    r'#?\w*makangratisgibran\w*|'
    r'#?\w*makanangratisgibran\w*|'
    r'#?\w*makangratisprabowogibran\w*|'
    r'#?\w*mkn[\s-]+gratis[\s-]+prabowo\w*|'
    r'#?\w*mkangratisprabowo\w*|'
    r'#?\w*mkngratisprabowo\w*|'
    r'#?\w*omprengbergizi\w*|'
    r'#?\w*menumbg\w*|'
    r'#?\w*dapursppg\w*|'
    r'#?\w*badan[\s-]+gizi[\s-]+nasional\w*|'
    r'\bbgn\b|#\w*bgn\w*|'
    r'dadan\s*hindayana|dadan\s*cs\b|(?:pak\s+dadan|kasus\s+dadan)\b|\bdadan\b.{0,40}(?:bgn|mbg|korupsi|tersangka|kejagung|sitaan|timses|kasus|gizi|dijerat)|'
    r'\bsppg\b|#\w*sppg\w*|'
    r'satuan\s*pelayanan\s*(?:pemenuhan\s*)?gizi|'
    r'dapur\s*(?:gizi|mbg|sehat|komunal|sppg)|'
    r'makan(?:an)?[\s-]+bergizi[\s-]+(?:bangun[\s-]+negeri|untuk[\s-]+indonesia)|'
    r'm\s*aling[,\s]+b\s*erkedok[,\s]+g\s*izi|'
    r'maling\s+berkedok\s+gizi|'
    r'em\s+be\s+ge|'
    r'murderer\s+by\s+government|badan\s+genosida\s+negara|'
    r'(?:prabowo|gibran).{0,60}(?:makan|mkn|gizi|susu|ompreng|bgn)|'
    r'(?:makan(?:an)?|mkn)[\s-]+gratis.{0,60}(?:prabowo|gibran|jokowi|presiden|pemerintah|heru\s*budi)',
    re.I
)

pat_mbg_cooccur = re.compile(
    r'(?:keracunan|beracun|makan(?:an)?\s+basi|makan\s+beracun).{0,250}(?:siswa|murid|guru|anak|sekolah|sd\b|smp\b|sma\b|santri|pesantren|pelajar|program|gratis|uji\s*coba|massal|katering|dapur|vendor|ompreng|korban)|'
    r'(?:siswa|murid|guru|anak|sekolah|santri|pesantren|pelajar|warga|korban).{0,250}(?:keracunan|beracun|makan(?:an)?\s+basi|makan\s+beracun)|'
    r'ompreng.{0,60}(?:makan|lauk|nasi|sayur|cuci|dapur|sekolah|anak|siswa|murid|guru|katering|program|sticker|stiker|relawan)|'
    r'(?:makan|cuci|wadah|tempat|piring|katering|dapur|guru|siswa|relawan).{0,60}ompreng|'
    r'susu\s+ikan|'
    r'susu.{0,50}(?:mbg|bgn|gratis|program|sekolah|siswa|anak|murid|pengganti\s+susu|impor|peternak|sapi|lokal)|'
    r'(?:menu|porsi|anggaran|biaya|alokasi).{0,50}(?:10\s*rb|15\s*rb|sepuluh\s*ribu|lima\s*belas\s*ribu|71\s*t|triliun|335\s*t|apbn|per\s+porsi|per\s+anak)|'
    r'(?:program|uji\s*coba|katering|vendor).{0,50}(?:10\s*rb|15\s*rb|porsi|nasi)|'
    r'(?:makan(?:an)?\s+bergizi.{0,70}(?:program|kebijakan|pemerintah|apbn|anggaran|alokasi|distribusi|uji\s*coba|prasmanan|periode|ompreng|dapur|katering|vendor|keracunan|basi|embel[-\s]*embel|disdik|disdikbud|kemendik|bgn|sppg))|'
    r'(?:(?:program|kebijakan|pemerintah|apbn|anggaran|alokasi|distribusi|uji\s*coba|prasmanan|periode|ompreng|dapur|katering|vendor|keracunan|basi|embel[-\s]*embel|disdik|disdikbud|kemendik|bgn|sppg).{0,70}makan(?:an)?\s+bergizi)|'
    r'(?:program|proyek|kebijakan|janji|uji\s*coba|anggaran|subsidi).{0,40}(?:makan\s+siang|makan(?:an)?\s+bergizi|makan\s+gratis|mkn\s+gratis)|'
    r'makan\s+siang.{0,50}(?:gratis|sekolah|siswa|murid|anak\s+sd|anak\s+smp|presiden|pemerintah)|'
    r'(?:dikasih|bagi-bagi|program)\s+(?:makan|mkn)\s+gratis|'
    r'(?:makan|mkn)\s+gratis.{0,60}(?:anak\s+sekolah|anak\s+sd|anak\s+smp|anak\s+sma|siswa|murid|santri|pelajar|stunting|rakyat|apbn|pembangunan|periode|brasil)|'
    r'#?\w*(?:makangratis|makanangratis)\w*.{0,50}(?:prabowo|gibran|bgn|sppg|presiden|pemerintah|heru\s*budi)|'
    r'(?:prabowo|gibran|bgn|sppg|presiden|pemerintah|heru\s*budi).{0,50}#?\w*(?:makangratis|makanangratis)\w*|'
    r'(?:sekolah|pendidikan)\s+gratis.{0,80}(?:makan(?:an)?\s+gratis|mbg|bgn)|'
    r'(?:makan(?:an)?\s+gratis|mbg|bgn).{0,80}(?:sekolah|pendidikan)\s+gratis|'
    r'(?:presiden|pemerintah|kemenkes|bgn|sppg).{0,60}(?:makanan?\s+bergizi|asupan\s+gizi|gizi\s+anak)|'
    r'semangka.{0,60}(?:setipis\s+hvs|beracun|mbg)',
    re.I
)

pat_sr_core = re.compile(
    r'#?\w*sekolah[-\s]*rakyat\w*|'
    r'#?\w*sekolahrakyat\w*|'
    r'#?\w*sekolah_rakyat\w*|'
    r'#?\w*sekolah[-\s]*gratis[-\s]*rakyat\w*|'
    r'#?\w*sekolah[-\s]*rakyat[-\s]*gratis\w*|'
    r'#?\w*pendidikan[-\s]*sekolah[-\s]*rakyat\w*|'
    r'#?\w*sekolah[-\s]*negeri[-\s]*rakyat\w*|'
    r'#?\w*sekolah[-\s]*rakyat[-\s]*berasrama\w*|'
    r'#?\w*program[-\s]*sekolah[-\s]*rakyat\w*|'
    r'#?\w*dukungsekolahrakyat\w*|'
    r'#?\w*pppksekolahrakyat\w*|'
    r'#?\w*(?:srt|srmp|srma|srd)\s*\d+|'
    r'#?\w*revitalisasi[-\s]*sekolah\w*|'
    r'sekolah\s+nasional\s+terintegrasi|\bsnt\b|'
    r'sekolah\s+swasta\s+gratis|'
    r'sekolah\s+gratis\s+prabowo|'
    r'(?:siswa|murid|guru|asrama)\s+sr\b',
    re.I
)

pat_sr_cooccur = re.compile(
    r'(?:sekolah|pendidikan)\s+gratis.{0,80}(?:rakyat|miskin|prabowo|gibran|pemerintah|program|asta\s*cita|asrama|daerah\s+terpencil|3t\b|anggaran|apbn|swasta|negeri|berkualitas|rempang|anak)|'
    r'(?:prabowo|gibran|pemerintah|presiden|gubernur|bupati|walikota).{0,80}(?:sekolah\s+gratis|sekolah\s+rakyat|pendidikan\s+gratis|revitalisasi\s+sekolah|sekolah\s+unggul|institusi\s+pendidikan|layanan\s+dasar.{0,30}sekolah|biaya\s+pendidikan|menggratiskan)|'
    r'kemendikdasmen|mendikdasmen|'
    r'(?:abdul\s+mu\'ti|mu\'ti).{0,70}(?:sekolah|pendidikan|kurikulum|siswa|murid|guru|spp|naik\s+kelas|kemendikdasmen|dewan\s+pendidikan)|'
    r'kemensos.{0,50}(?:sekolah|pendidikan|asrama)|'
    r'sekolah\s+berasrama.{0,60}(?:gratis|rakyat|pemerintah|prabowo|kemensos|ancaman)|'
    r'dana\s+bos|bantuan\s+operasional\s+sekolah|'
    r'(?:biaya|uang)\s+(?:sekolah|spp|gedung|seragam).{0,50}(?:gratis|bebas|rakyat|pemerintah)|'
    r'(?:program|evaluasi|anggaran).{0,70}(?:sekolah\s+rakyat|pendidikan\s+rakyat|revitalisasi\s+sekolah)|'
    r'beasiswa\s+afirmasi|kip\s+kuliah.{0,50}(?:3t\b|daerah\s+terpencil|putus\s+sekolah|kuota)|'
    r'putus\s+sekolah.{0,60}(?:program|bantuan|kemiskinan|saber\s+do|pemprov|pemkab|prabowo)|'
    r'(?:sekolah|pendidikan)\s+gratis.{0,80}(?:makan(?:an)?\s+gratis|mbg|bgn)|'
    r'(?:makan(?:an)?\s+gratis|mbg|bgn).{0,80}(?:sekolah|pendidikan)\s+gratis',
    re.I
)

pat_kdmp_core = re.compile(
    r'#?\b(?:kdmp|kdkmp|knmp)\b|'
    r'#\w*(?:kdmp|kdkmp|knmp)\w*|'
    r'#?\bagrinas\b|#\w*agrinas\w*|'
    r'#?\w*koperasidesa\w*|#?\w*koperasidesamerahputih\w*|'
    r'#?\w*koprasimerahputih\w*|'
    r'#?\w*koperasimerahputih\w*|'
    r'#?\w*koprasidesa\w*|'
    r'(?:koperasi|koprasi)[-\s]+desa[-\s]+merah[-\s]+putih|'
    r'(?:koperasi|koprasi)[-\s]+merah[-\s]+putih|'
    r'desa[-\s]+merah[-\s]+putih|'
    r'(?:koperasi|koprasi)[-\s]+desa|'
    r'kopdes[-\s]+merah[-\s]+putih|'
    r'#?\w*kopdes\b|#\w*kopdes\w*|'
    r'#?\w*manajerkopdes\w*|#?\w*managerkdmp\w*|'
    r'di\s*kdkmp|dikdkmp|di\s*kdmp|dikdmp',
    re.I
)

pat_kdmp_cooccur = re.compile(
    r'(?:koperasi|koprasi|kopdes|bumdes).{0,90}(?:merah[-\s]+putih|prabowo|program|pangan|distribusi|petani|nelayan|mbg|dana\s+desa|modal|umkm|bumn|himbara|kredit|sindikasi|plafon|triliun|miliar|manajer|rekrutmen|panselnas|phtc|budi\s+arie|ferry\s+juliantono|kemenkop|gaji\s+asn|gaji\s+pns|penyaluran\s+gaji|layanan\s+gaji|pupuk\s+subsidi|gas\s+3\s*kg|pinjol|inpres|instruksi\s+presiden|lokasi|gedung|modern)|'
    r'(?:gaji\s+asn|gaji\s+pns|penyaluran\s+gaji).{0,90}(?:koperasi|koprasi|kemenkop|kopdes)|'
    r'kemenkop.{0,90}(?:desa|koperasi|umkm|asn|pns|pangan|ai|digital)|'
    r'menteri\s+koperasi|kementerian\s+koperasi|'
    r'menko\s+pangan.{0,90}(?:koperasi|desa|pangan|distribusi)|'
    r'(?:program|proyek|kebijakan|alokasi).{0,90}(?:koperasi\s+desa|desa\s+mandiri\s+pangan|kdmp|kdkmp|knmp|kopdes)|'
    r'(?:koperasi|koprasi)\s+umkm|'
    r'dana\s+desa.{0,70}(?:koperasi|bumdes|ketahanan\s+pangan|pertumbuhan\s+ekonomi|blt)',
    re.I
)

# 3. Pola Noise Negatif (Dua Tingkat: Fatal Noise vs Context Noise)
# 3a. Fatal Noise: Selalu dibuang tanpa kompromi (Spam teknis, curhat hamil/laktasi, CSR privat non-pemerintah, promo ultah resto, isu asing)
pat_fatal_noise = re.compile(
    r"it\s+looks\s+like\s+you\s+were\s+misusing|you['’]ve\s+been\s+temporarily\s+blocked|"
    r'pejuang\s+garis\s+dua|promil|senam\s+hamil|yoga\s+hamil|sufor|pompa\s+asi|kantung\s+asi|kantong\s+asi|usg\s+(?:2d|3d|4d)|bbj\s+janin|periksa\s+kandungan|'
    r'sekolah\s+anak\s+percaya|andien\s+aisyah|sekolah\s+koding|sekolah\s+coding|sekolah\s+forex|sekolah\s+minggu\b|'
    r'makan\s+gratis\s+rp\s*0|promo\s+ultah.{0,30}makan|syarat\s+ketentuan.{0,30}makan\s+gratis|klaim\s+promo.{0,30}makan\s+gratis|'
    r'manchester\s+united|tottenham|mbeumo|solanke|geng\s+motor|savado|'
    r'racun\s+shopee|racun\s+tiktok|jasa\s+gestun|running\s+threads|lari\s+setiap\s+hari|freedive|'
    r'kpm\s+dan\s+kkm|isu\s+jerebu|bakar\s+hutan\s+korang|menteri\s+malaysia|kerajaan\s+malaysia|sekolah\s+kebangsaan\s+rakyat|sk\s+rakyat\b|pulau\s+pinang|tarikh|timbalan|icu\s+jpm',
    re.I
)

# 3b. Context Noise: Disaring HANYA jika postingan tidak memiliki entitas kebijakan inti kuat (has_strong_core == False)
# Jika postingan memuat MBG / BGN / SPPG / Sekolah Rakyat / KDMP / Dadan Hindayana, postingan tetap AMAN/DISIMPAN!
pat_context_noise = re.compile(
    r'rumah\s+makan\s+gratis|warung\s+makan\s+gratis|nasi\s+darurat|makanan\s+darurat|sedekah\s+makan|traktir\s+makan|bayar\s+seikhlasnya|jumat\s+berkah.{0,30}(?:makan|bagi|nasi)|'
    r'sekolah\s+relawan|sekolah\s+master\b|sekolah\s+alam\b|'
    r'bawang\s+putih|kayu\s+putih|satelit\s+merah\s+putih|'
    r'butuh\s+dana\s+darurat|pinjam\s+di\s+koperasi|pinjaman\s+online|'
    r'kang\s+dedi\s+mulyadi|\bdedi\s+mulyadi\b|\bkdm\b.{0,30}(?:gubernur|jabar|bapakaing|rembang|purwakarta|persikas)|'
    r'naik\s+ompreng|mobil\s+ompreng|'
    r'beli\s+2\s+gratis\s+1|promo\s+diskon|beli\s+susu\s+di\s+(?:indomaret|alfamart)|'
    r'polemik\s+ijazah|ijazah\s+(?:jokowi|gibran|palsu)|ijazahnya\s+gak\s+ada|surat\s+suara\s+paslon|demo\s+di\s+perancis|maling\s+motor|kopassus|'
    r'saham\s+bbca|laba\s+q1|jerebu|'
    r'tips\s+parenting|tips\s+ibu|bekal\s+anak\s+sekolah|resep\s+bekal|isi\s+piringku|gizi\s+seimbang.{0,30}(?:balita|batita|anak)|sarapan\s+pagi.{0,30}(?:sehat|bergizi)|lagu\s+anak\b|kebiasaan\s+anak\s+indonesia',
    re.I
)

# Platform yang akan difilter
PLATFORMS_TO_FILTER = ['Twitter', 'Instagram', 'TikTok', 'YouTube', 'Facebook', 'News']


def run_filter_pipeline():
    print("=" * 80)
    print("MASTER DATA CLEANSING (TWO-TIERED NOISE ARCHITECTURE: FATAL SPAM VS CONTEXT NOISE)")
    print("=" * 80)

    # 1. Baca Konfigurasi Keywords dari cell_02_config.py
    print(f"Membaca konfigurasi keyword dari: {CONFIG_PATH}...")
    topics_dict = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            cfg_code = f.read()
        exec(compile(cfg_code, CONFIG_PATH, "exec"), {"__file__": CONFIG_PATH, "os": os, "re": re}, topics_dict)
        TOPICS = topics_dict.get("TOPICS", {})
        print(f"Berhasil membaca {len(TOPICS)} topik dengan total {sum(len(v) for v in TOPICS.values())} keywords.")
    except Exception as e:
        print(f"[WARN] Gagal membaca cell_02_config.py: {e}. Menggunakan konfigurasi standar.")
        TOPICS = {}

    all_discarded_urls = set()
    all_discarded_contents = set()

    for plat in PLATFORMS_TO_FILTER:
        fname = f'scraped_{plat}_Only.csv'
        fpath = os.path.join(SCRAPE_DIR, fname)
        backup_fname = f'scraped_{plat}_Only_BEFORE_FILTER.csv'
        backup_path = os.path.join(SCRAPE_DIR, backup_fname)

        # Selalu baca dari sumber murni BEFORE_FILTER jika ada
        read_path = backup_path if os.path.exists(backup_path) else fpath
        if not os.path.exists(read_path):
            continue

        # Buat backup jika belum ada
        if not os.path.exists(backup_path) and os.path.exists(fpath):
            shutil.copy2(fpath, backup_path)
            print(f"\n[BACKUP] Menyimpan data mentah awal ke: {backup_fname}")

        df_raw = pd.read_csv(read_path, low_memory=False)
        total_raw = len(df_raw)

        col = 'content' if 'content' in df_raw.columns else 'description'
        content_raw = df_raw[col].fillna('').astype(str)

        # Evaluasi gabungan: Jika ada kolom description (pada News & YouTube), gabungkan keduanya untuk pencocokan topik
        if 'description' in df_raw.columns and 'content' in df_raw.columns:
            eval_raw = content_raw + ' ' + df_raw['description'].fillna('').astype(str)
        else:
            eval_raw = content_raw

        # Normalisasi teks komprehensif: Unicode NFKD (untuk font tebal/miring/fancy sosmed), HTML unescape, dan spasi khusus
        def normalize_text_for_eval(text):
            t = html.unescape(str(text))
            t = unicodedata.normalize('NFKD', t)
            t = re.sub(r'[\u00a0\u200b\u200c\u200d\u202f\ufeff]', ' ', t)
            return t

        content = eval_raw.apply(normalize_text_for_eval)

        # Aturan Reply: Pada Instagram, TikTok, YouTube, Facebook, News, mention di awal (^@) adalah tagar akun mitra/resmi, bukan reply!
        if plat in ['Instagram', 'TikTok', 'YouTube', 'Facebook', 'News']:
            reply_pat = re.compile(
                r'(?:^|[\r\n\s])(?:Membalas\s+@|Replying\s+to\s+@|Balasan\s+pada\s+postingan|Balasan\s+untuk\s+@)',
                re.I
            )
        else:
            reply_pat = re.compile(
                r'(?:^|[\r\n\s])(?:Membalas\s+@|Replying\s+to\s+@|Balasan\s+pada\s+postingan|Balasan\s+untuk\s+@)|(?:^@[\w\.-]+\s+)',
                re.I
            )

        # Deteksi akun resmi unit pelayanan gizi (SPPG) di sosmed (TikTok & Instagram)
        is_sppg_account = pd.Series(False, index=df_raw.index)
        if 'author' in df_raw.columns:
            is_sppg_account = is_sppg_account | df_raw['author'].fillna('').astype(str).str.contains(r'\bsppg\b', case=False, regex=True)
        if 'handle' in df_raw.columns:
            is_sppg_account = is_sppg_account | df_raw['handle'].fillna('').astype(str).str.contains(r'sppg', case=False, regex=True)

        is_reply = content.str.contains(reply_pat)
        match_mbg = content.str.contains(pat_mbg_core) | content.str.contains(pat_mbg_cooccur) | is_sppg_account
        match_sr = content.str.contains(pat_sr_core) | content.str.contains(pat_sr_cooccur)
        match_kdmp = content.str.contains(pat_kdmp_core) | content.str.contains(pat_kdmp_cooccur)

        has_any_topic = match_mbg | match_sr | match_kdmp
        has_strong_core = (
            content.str.contains(pat_mbg_core) |
            content.str.contains(pat_sr_core) |
            content.str.contains(pat_kdmp_core) |
            is_sppg_account
        )

        hits_fatal = content.str.contains(pat_fatal_noise)
        hits_context = content.str.contains(pat_context_noise)
        # Context noise hanya aktif jika postingan tidak memiliki entitas kebijakan inti kuat
        hits_noise = hits_fatal | (hits_context & (~has_strong_core))

        # Postingan dengan has_strong_core tetap disimpan meskipun diawali mention, tetapi jika terdeteksi spam fatal/noise maka difilter
        keep_mask = ((~is_reply) | has_strong_core) & has_any_topic & (~hits_noise)
        discard_mask = ~keep_mask

        # Rekonsiliasi Topik (Cross-topic alignment)
        new_topic = df_raw['topic'].copy() if 'topic' in df_raw.columns else pd.Series('General', index=df_raw.index)
        is_actually_mbg = (new_topic != 'MBG') & match_mbg & (~match_kdmp) & (~match_sr)
        new_topic[is_actually_mbg] = 'MBG'

        is_actually_kdmp = (new_topic != 'KDMP') & match_kdmp & (~match_mbg) & (~match_sr)
        new_topic[is_actually_kdmp] = 'KDMP'

        is_actually_sr = (new_topic != 'Sekolah_Rakyat') & match_sr & (~match_mbg) & (~match_kdmp)
        new_topic[is_actually_sr] = 'Sekolah_Rakyat'

        df_raw['topic'] = new_topic

        df_clean = df_raw[keep_mask].copy()
        df_discarded = df_raw[discard_mask].copy()

        # Buat alasan pemindahan yang transparan
        reasons = []
        for idx in df_discarded.index:
            if is_reply[idx] and not has_strong_core[idx]:
                reasons.append('Postingan Balasan / Reply (Membalas @... / kehilangan konteks)')
            elif hits_fatal[idx]:
                reasons.append('Spam Fatal / Entitas Asing Terdeteksi (Curhat Laktasi/Sekolah Swasta CSR/Promo Resto/Isu Luar Negeri)')
            elif hits_context[idx] and not has_strong_core[idx]:
                reasons.append('Noise Kontekstual Tanpa Kebijakan Inti (Amal Mandiri/Pinjol/Kuliner Non-Program)')
            else:
                reasons.append('Rekomendasi Algoritmik / Konten Nyasar (Tanpa kata kunci & sentimen topik)')

        df_discarded['alasan_rekomendasi_hapus'] = reasons

        # Simpan kembali
        df_clean.to_csv(fpath, index=False, encoding='utf-8')
        kandidat_path = os.path.join(SCRAPE_DIR, f'scraped_{plat}_Kandidat_Hapus.csv')
        df_discarded.to_csv(kandidat_path, index=False, encoding='utf-8')

        # Catat URL dan text discarded untuk menyaring master parquet
        if 'url' in df_discarded.columns:
            valid_urls = df_discarded['url'].dropna().astype(str).str.strip()
            all_discarded_urls.update(u for u in valid_urls if u and u != 'nan')
        all_discarded_contents.update(df_discarded[col].dropna().astype(str).str.strip())

        # Sync ke folder cobaAPI/ScrapeData jika ada
        if os.path.exists(COBA_SCRAPE_DIR):
            shutil.copy2(fpath, os.path.join(COBA_SCRAPE_DIR, fname))
            shutil.copy2(kandidat_path, os.path.join(COBA_SCRAPE_DIR, f'scraped_{plat}_Kandidat_Hapus.csv'))

        print(f"[{plat:10}] Awal: {total_raw:>6,} | Bersih: {len(df_clean):>6,} ({len(df_clean)/total_raw*100:5.1f}%) | Kandidat Hapus: {len(df_discarded):>5,}")

    print("\n" + "=" * 80)
    print("🚀 SINKRONISASI KE DATASET MASTER ENRICHED CORPUS")
    print("=" * 80)

    PARQUET_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
    CSV_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.csv')
    MASTER_PATH = os.path.join(DATA_DIR, 'master_scraped_corpus.csv')
    PARQUET_BAK = os.path.join(DATA_DIR, 'enriched_corpus_for_sna_BEFORE_FILTER.parquet')

    # Selalu baca dari backup murni BEFORE_FILTER agar data riil yang diselamatkan masuk kembali
    source_parquet = PARQUET_BAK if os.path.exists(PARQUET_BAK) else PARQUET_PATH

    if os.path.exists(source_parquet):
        print(f"Membaca {os.path.basename(source_parquet)}...")
        df_p = pd.read_parquet(source_parquet)
        total_p = len(df_p)

        # Identifikasi baris yang harus dibuang
        url_mask = df_p['url'].astype(str).str.strip().isin(all_discarded_urls)
        content_raw_mask = df_p['raw_content'].astype(str).str.strip().isin(all_discarded_contents)
        discard_in_p = url_mask | content_raw_mask

        df_p_clean = df_p[~discard_in_p].copy()
        print(f"Master Parquet: {total_p:,} -> {len(df_p_clean):,} baris (Dibersihkan {discard_in_p.sum():,} baris noise sesungguhnya)")

        # Simpan parquet bersih
        df_p_clean.to_parquet(PARQUET_PATH, engine='pyarrow', index=False)
        print(f"💾 Tersimpan ke: {PARQUET_PATH}")

        # Simpan CSV bersih jika ada
        if os.path.exists(CSV_PATH):
            df_p_clean.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
            print(f"💾 Tersimpan ke: {CSV_PATH}")

    if os.path.exists(MASTER_PATH):
        print("Membaca master_scraped_corpus.csv...")
        df_m = pd.read_csv(MASTER_PATH, low_memory=False)
        m_url = df_m['url'].astype(str).str.strip().isin(all_discarded_urls)
        m_content = df_m['raw_content'].astype(str).str.strip().isin(all_discarded_contents) if 'raw_content' in df_m.columns else False
        df_m_clean = df_m[~(m_url | m_content)].copy()
        df_m_clean.to_csv(MASTER_PATH, index=False, encoding='utf-8-sig')
        print(f"💾 Master corpus diperbarui: {len(df_m):,} -> {len(df_m_clean):,} baris")

    print("\n✅ BERHASIL MEMPERBARUI SELURUH DATASET DENGAN ARSITEKTUR TWO-TIERED NOISE & KEBIJAKAN INTI!")


if __name__ == "__main__":
    run_filter_pipeline()
