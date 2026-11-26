"""
Engine Pemilah Relevansi Konteks Multi-Lapis — LEMBAGA ISLAM INDONESIA (LII)
==========================================================================
Menyaring postingan di luar konteks penelitian (3 topik: Nahdlatul_Ulama,
Muhammadiyah, BAZNAS) di seluruh 7 platform media sosial & berita.

Lapisan Deteksi:
  0. Normalisasi Unicode NFKD (menangani font dekoratif/bold/italic media sosial)
  1. Gate Aksara & Bahasa Asing (non-Latin, Rumania/Belanda/Swedia/Portugis/Inggris/Melayu MY)
  2. Gate Bahasa Sunda Pembajak Kata 'nu' ("yang")
  3. Gate Homonim Non-LII ('NU: Carnival', 'NU Sentral', 'NU Cebu/Manila', 'NU Metal', 'Aisyah', iklan/judol/spam)
  4. Jangkar Entitas & Konteks Topik Bertingkat (Tier-A Eksplisit & Tier-B Ko-okurensi Domain)
  5. Rekonsiliasi Lintas-Topik bila teks eksplisit hanya membahas lembaga LII lain
"""

import re
import unicodedata

# ==============================================================================
# 1. POLA AKSARA & BAHASA ASING
# ==============================================================================
_RE_FOREIGN_SCRIPT_NO_LATIN_MIX = re.compile(
    r'[\u0E00-\u0E7F'          # Thai
    r'\u0B80-\u0BFF'           # Tamil
    r'\u0900-\u097F'           # Devanagari
    r'\u0A00-\u0A7F'           # Gurmukhi (Punjabi)
    r'\u0400-\u04FF'           # Cyrillic
    r'\u0370-\u03FF'           # Greek
    r'\uAC00-\uD7AF\u1100-\u11FF'  # Korean Hangul
    r'\u3040-\u30FF\u4E00-\u9FFF]' # Japanese / CJK
)
_RE_ARABIC_CHARS = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
_RE_VIET_DIACRITICS = re.compile(r'[ăâđêôơưĂÂĐÊÔƠƯảãáàạấầẩẫậắằẳẵặẻẽéèẹếềểễệỉĩíìịỏõóòọốồổỗộớờởỡợủũúùụứừửữựỷỹýỳỵ]')

# Bahasa Eropa yang memakai kata "nu" ("not" / "now" / "naked") atau mirip
_RE_ROMANIAN = re.compile(
    r'\b(?:de\s+ce\s+nu|nu\s+mai|nu\s+vreau|nu\s+am|nu\s+e|nu\s+pot|nu\s+stiu|nu\s+știu|'
    r'nu\s+poate|care\s+nu|pentru|dar\s+nu|dacă|daca\s+nu|atunci\s+cand|păi|'
    r'țară|tara|românia|romania|mulțumesc|mulțumim|nicuşor|nicușor|băsescu)\b',
    re.I,
)
_RE_DUTCH = re.compile(
    r'\b(?:ook\s+nu|nu\s+ook|wat\s+een|volgens\s+jullie|het\s+is|het\s+kan|'
    r'tweede\s+kamer|niet|maar\s+de|van\s+de|met\s+de|voor\s+het|deze|nederland|'
    r'amsterdam|antwerpse|beschieten|defensie|twee|heeft|zullen|omdat)\b',
    re.I,
)
_RE_SWEDISH = re.compile(
    r'\b(?:uppäten|det\s+är|jag\s+har|inte\s+nu|nu\s+är|och\s+det|för\s+att|med\s+en)\b',
    re.I,
)
_RE_PORTUGUESE_SPANISH = re.compile(
    r'\b(?:caralho|muito\s+feliz|muito|você|voce|não|nao|obrigado|obrigada|para\s+o|'
    r'como\s+estás|gracias|también|tambien|entonces)\b',
    re.I,
)
_RE_MALAY_GOV = re.compile(
    r'\b(?:kerajaan\s+(?:malaysia|negeri)|menteri\s+malaysia|penjawat\s+awam|'
    r'negeri\s+(?:perak|selangor|sabah|sarawak|johor|kedah|kelantan|terengganu|pahang|melaka|perlis)|'
    r'putrajaya|yadim|jakim|zakat\s+selangor|lembaga\s+zakat\s+negeri|'
    r'ringgit|rm\s*\d+(?:[.,]\d+)?)\b',
    re.I,
)

# Stopword Bahasa Indonesia & Bahasa Inggris untuk rasio bahasa
_RE_ID_STOPWORDS = re.compile(
    r'\b(?:yang|dan|di|ke|dari|ini|itu|untuk|dengan|pada|akan|tidak|tak|bukan|atau|juga|'
    r'sudah|belum|karena|kalau|jika|agar|supaya|sebagai|oleh|dalam|atas|bisa|dapat|harus|'
    r'ada|adalah|mereka|kita|kami|saya|aku|kamu|dia|banget|saja|aja|kok|sih|dong|nih|'
    r'yg|dg|dgn|utk|bgt|gak|ga|nggak|tdk|sdh|blm|krn|klo|dlm|buat|sama|lagi|mau|'
    r'orang|warga|masyarakat|pemerintah|indonesia|umat|islam|muslim|ketua|pengurus|'
    r'jadi|titik|awal|penguatan|kaderisasi|abad|kedua|bupati|walikota|gubernur|kabupaten|'
    r'kecamatan|provinsi|daerah|nasional|apresiasi|dorong|soroti|buka|gelar|siap|'
    r'kepengurusan|langkah|melangkah|pascamuktamar|pramuktamar|pramunas|jatim|jateng|jabar)\b',
    re.I,
)
_RE_EN_STOPWORDS = re.compile(
    r'\b(?:the|and|that|this|with|from|you|your|are|was|were|have|has|had|what|when|'
    r'where|which|they|their|about|because|would|could|should|will|been|being|'
    r'more|most|than|into|only|very|just|over|also|such|how|why|who|not|but)\b',
    re.I,
)

# ==============================================================================
# 2. GATE BAHASA SUNDA PEMBAJAK KATA "nu" ("yang")
# ==============================================================================
_RE_SUNDA_MARKERS = re.compile(
    r'\b(?:naon|kieu|ieu|eta|eweuh|euweuh|henteu|teu|kanggo|kanggé|pisan|'
    r'cenah|saurna|ceuk|ayena|ayenamah|ayeuna|kamari|baheula|nepika|borojol|'
    r'daek|hoyong|hirup|maot|paeh|abdi|abi|aing|kumaha|iraha|'
    r'jeung|maneh|kasep|geulis|goreng|nyaho|ngaran|pikeun|hungkul|bade|moal|'
    r'sudi|kadieu|kaditu|nuju|beuki|hapunteun|hapunten|deui|'
    r'oge|ogé|nyarios|sakedik|sakitu|sadayana|sawios|tiasa|teuing|atuh)\b',
    re.I,
)
_RE_SUNDA_NU_GRAMMAR = re.compile(
    r'(?<![A-Za-z])(?:nu\s+(?:kieu|aya|maot|hirup|geulis|kasep|alus|goreng|bener|salah|penting|'
    r'lain|mana|saha|nuju|keur|hade|nyaho|ngaran|kamari|ayeuna|kudu|bisa|hayang|'
    r'resep|bogoh|saat|eta|ieu|ditanya|dibahas|nyieun|mikir|datang|indit|ngawaro)|'
    r'(?:ku|ti|ka|eweuh|euweuh|aya|saha|naon|mana|ieu|eta|lamun|sabab|teh|mah)\s+nu)(?![A-Za-z])'
)

# ==============================================================================
# 3. HOMONIM NON-LII & SPAM / PROMOSI TIDAK RELEVAN
# ==============================================================================
_RE_NU_HOMONYMS = re.compile(
    r'\b(?:nu[\s:\-]+(?:carnival|karnival|metal|disco|jazz|soul|folk|wave|skin|'
    r'cebu|manila|bacolod|laguna|fairview|moa|nazareth|bulldogs|sentral|raw|build))\b|'
    r'nu[\'\u2019`]?est\b|\bnuskin\b|\bnunazarethschool\b|'
    r'\b(?:national\s+university|northeastern\s+university|northwestern\s+university|'
    r'niagara\s+university|pep\s+squad|uaap\s+musim|uaap\s+season|dramatical\s+murder)\b',
    re.I,
)
_RE_AISYAH_PERSON = re.compile(r'\baisyah\b', re.I)
_RE_OTHER_HOMONYMS = re.compile(
    r'\b(?:immigration|immortal|immutable|immune|assets\s+under\s+management|yoga\s+aum|om\s+aum)\b',
    re.I,
)
_RE_SPAM_ADS = re.compile(
    r'\b(?:slot\s+gacor|maxwin|rtp\s+live|togel|open\s+bo|vcs\s+murah|racun\s+shopee|'
    r'keranjang\s+kuning|jasa\s+gestun|perpanjang\s+stnk|dijual\s+tanah|dijual\s+rumah|'
    r'jual\s+mobil|sewa\s+apartemen|it\s+looks\s+like\s+you\s+were\s+misusing|'
    r'you[\'\u2019`]ve\s+been\s+temporarily\s+blocked)\b|'
    r'^(?:wts|wtb|wtt)\b',
    re.I,
)

# ==============================================================================
# 4. ENTITAS & KONTEKS BERTINGKAT PER TOPIK LII
# ==============================================================================
# --- A. NAHDLATUL ULAMA (NU) ---
_RE_NU_TIER_A = re.compile(
    r'\b(?:nah(?:dl|dh|d)[ao]tul\s*ulama|nah(?:dl|dh|d)[ao]tul|nahdliyin|nahdliyyin|nahdliyyah|'
    r'pbnu|pwnu|pcnu|mwcnu|prnu|pcinu|rminu|ldnu|lpnu|isnu|lakpesdam|pagar\s*nusa|'
    r'gp\s*ansor|ansor|banser|muslimat\s*nu|fatayat\s*nu|ipnu|ippnu|pmii|kmnu|'
    r'lazisnu|nu\s*care|nu\s*online|tvnu|tv\s*nu|jatman|lbm[\s\-]*nu|rm[\s\-]*nu|'
    r'ma[\'\u2019`]?arif\s+nu|unusia|unusa|unugiri|unuja|unugha|unusida|unublitar|unucirebon|'
    r'tebuireng|lirboyo|ploso|krapyak|ahl[ui]s?sun+ah\s+wal\s*jama[\'\u2019`]?ah|aswaja|islam\s+nusantara|ulama\s+nusantara|ala\s+nu|'
    r'rais\s*aam|ro[\'\u2019`]?is\s*aam|katib\s*aam|tanfidziyah|syuriah|syuriyah|'
    r'muktamar\s+nu|pascamuktamar|pramuktamar|harlah\s+nu|munas\s+nu|konbes\s+nu|khittah\s+nu|'
    r'nu\s+peduli|nu\s+fest|nufest|'
    r'ormas\s+nu|warga\s+nu|kader\s+nu|tokoh\s+nu|banom\s+nu|pengurus\s+nu|ketum\s+nu|'
    r'mwc\s+nu|pc\s+nu|pw\s+nu|pr\s+nu|rs\s*nu|rsnu|jam[\'\u2019`]?iyah\s+nu|'
    r'hasyim\s+asy[\'\u2019`]?ari|gus\s+dur|abdurrahman\s+wahid|gus\s+yahya|'
    r'yahya\s+cholil|said\s+aqil|miftachul\s+akhyar|gus\s+mus|gus\s+baha)\b|'
    r'[#@][a-z0-9_.]*(?:nah(?:dl|dh|d)[ao]tul\w*|nahdliyin\w*|pbnu|pwnu|pcnu|mwcnu|prnu|'
    r'bansernu|gpansor|ansorbanser|pagarnusa|muslimatnu|fatayatnu|lazisnu|harlahnu|muktamarnu|nuonline)[a-z0-9_]*',
    re.I,
)
_RE_NU_ISLAM_CONTEXT = re.compile(
    r'\b(?:ulama|kiai|kyai|kh\.|gus\s+\w+|pesantren|ponpes|santri|ustadz|ustad|habib|'
    r'ormas|muhammadiyah|ldii|salafi|wahabi|persis|al-irsyad|mui|kemenag|'
    r'islam|muslim|umat|masjid|mushola|ramadan|ramadhan|lebaran|idul\s*fitri|idul\s*adha|'
    r'hisab|rukyat|hilal|tahlil|tahlilan|maulid|sholawat|shalawat|istighosah|dakwah|'
    r'pengajian|kajian|madrasah|kitab\s+kuning|fiqih|fikih|fatwa|iftitah|madzhab|mazhab|'
    r'jam[\'\u2019`]?iyah|jama[\'\u2019`]?ah|muktamar|pascamuktamar|pramuktamar|'
    r'harlah|munas|pramunas|konferwil|konfercab|raker|rakernas|bahtsul\s+masail|kaderisasi|'
    r'tambang|konsesi|kpk|menteri|presiden|pemerintah|politik|pkb|pancasila|nkri|'
    r'toleransi|moderat|ketua|pengurus|banom|kader|ranting|cabang|wilayah|zakat|wakaf|'
    r'bupati|walikota|gubernur|berbangsa|bernegara|abad\s+kedua)\b',
    re.I,
)
_RE_BARE_NU_UPPER = re.compile(r'(?<![A-Za-z])NU(?![A-Za-z])')
_RE_BARE_NU_ANY = re.compile(r'\bnu\b', re.I)

# --- B. MUHAMMADIYAH ---
_RE_MUH_TIER_A = re.compile(
    r'\b(?:muhammad[iy]+ah|mukamadi[ay]+|aisyiyah|[\'\u2019`]aisyiyah|lazismu|'
    r'pemuda\s+muhammadiyah|ikatan\s+mahasiswa\s+muhammadiyah|ikatan\s+pelajar\s+muhammadiyah|'
    r'nasyiatul\s+aisyiyah|nasyiah|hizbul\s+wathan|tapak\s+suci|kokam|mdmc|tvmu|lensamu|'
    r'suara\s+muhammadiyah|majelis\s+tarjih|pku\s+muhammadiyah|rs\s+muhammadiyah|'
    r'rsu\s+aisyiyah|tk\s+aba|mudipat|amal\s+usaha\s+muhammadiyah|'
    r'pp\s+muhammadiyah|pimpinan\s+pusat\s+muhammadiyah|pimpinan\s+wilayah\s+muhammadiyah|'
    r'ahmad\s+dahlan|siti\s+walidah|nyai\s+walidah|haedar\s+nashir|abdul\s+mu[\'\u2019`]?ti|'
    r'abdul\s+muti|din\s+syamsuddin|syafii\s+maarif|amien\s+rais|busyro\s+muqoddas|'
    r'hisab\s+hakiki|wujudul\s+hilal|khgt|kalender\s+hijriah\s+global|'
    r'unismuh|uhamka|umsida|umsu)\b|'
    r'[#@][a-z0-9_.]*(?:muhammadiyah|aisyiyah|lazismu|pemudamuhammadiyah|immmuhammadiyah|lensamu|tvmu)[a-z0-9_]*',
    re.I,
)
_RE_MUH_SHORT_ORGS = re.compile(r'\b(?:aum|pwm|pdm|pcm|prm|imm|ipm|umy|ums|umm|uad|umj|ump)\b', re.I)
_RE_MUH_CONTEXT = re.compile(
    r'\b(?:muhammad[iy]+ah|aisyiyah|tarjih|hisab|hilal|tanwir|muktamar|milad|ortom|'
    r'pengajian|kajian|dakwah|islam|muslim|umat|masjid|sekolah|sdm|smpm|smam|kampus|'
    r'mahasiswa|kader|pimpinan|majelis|lembaga|persyarikatan|perserikatan|wakaf|zakat|infak)\b',
    re.I,
)

# --- C. BAZNAS ---
_RE_BAZNAS_TIER_A = re.compile(
    r'\b(?:baznas|baznaz|badan\s+amil\s+zakat(?:\s+nasional)?|bazda|'
    r'rumah\s+sehat\s+baznas|zmart|zchicken|zcoffee|z-auto|cendekia\s+baznas|'
    r'baznas\s+tanggap\s+bencana|noor\s+achmad|amil\s+zakat|lembaga\s+amil\s+zakat)\b|'
    r'[#@][a-z0-9_.]*(?:baznas|baznaz)[a-z0-9_]*',
    re.I,
)
_RE_UPZ = re.compile(r'\bupz\b', re.I)
_RE_ZAKAT_CORE = re.compile(
    r'\b(?:zakat(?:\s+(?:fitrah|mal|maal|profesi|penghasilan|produktif|nasional))?|'
    r'muzakki|muzaki|mustahik|mustahiq|8\s+asnaf|delapan\s+asnaf|infaq|infak|sedekah|fidyah)\b',
    re.I,
)
_RE_NON_BAZNAS_ZAKAT_ORG = re.compile(
    r'\b(?:lazismu|lazisnu|dompet\s+dhuafa|rumah\s+zakat|ybm\s+pln|dt\s+peduli|izi|inisiatif\s+zakat)\b',
    re.I,
)
_RE_RIVAL_TOPICS = {
    'Nahdlatul_Ulama': _RE_NU_TIER_A,
    'Muhammadiyah': _RE_MUH_TIER_A,
    'BAZNAS': _RE_BAZNAS_TIER_A,
}


def _count_matches(pattern, text):
    return len(pattern.findall(text))


def _strip_hashtags_mentions_urls(text):
    t = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    t = re.sub(r'[@#][\w\.-]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def _detect_foreign_or_noise(text_norm, topic_hint):
    """Mengecek bahasa asing, aksara asing, Sunda 'nu', homonim non-LII, dan spam."""
    has_any_tier_a = bool(
        _RE_NU_TIER_A.search(text_norm)
        or _RE_MUH_TIER_A.search(text_norm)
        or _RE_BAZNAS_TIER_A.search(text_norm)
    )
    id_count = _count_matches(_RE_ID_STOPWORDS, text_norm)

    # 1. Aksara non-Latin (Thai, Tamil, Korea, Jepang, CJK, Cyrillic, Punjabi) & Vietnam
    foreign_chars = len(_RE_FOREIGN_SCRIPT_NO_LATIN_MIX.findall(text_norm))
    if foreign_chars >= 4 and not (has_any_tier_a and id_count >= 2):
        return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks didominasi aksara non-Indonesia'
    if len(_RE_VIET_DIACRITICS.findall(text_norm)) >= 2 and not has_any_tier_a:
        return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks terdeteksi bahasa Vietnam'

    # 2. Aksara Arab murni tanpa kata Indonesia / tanpa entitas LII
    arabic_chars = len(_RE_ARABIC_CHARS.findall(text_norm))
    if arabic_chars >= 15 and id_count == 0 and not has_any_tier_a:
        return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks murni berbahasa Arab tanpa konteks LII Indonesia'

    # 3. Bahasa Eropa / Melayu pembajak kata 'nu' atau keyword lokal
    if not has_any_tier_a:
        if _count_matches(_RE_ROMANIAN, text_norm) >= 1 and id_count <= 2:
            return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks terdeteksi bahasa Rumania (kata "nu")'
        if _count_matches(_RE_DUTCH, text_norm) >= 2 and id_count <= 2:
            return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks terdeteksi bahasa Belanda (kata "nu")'
        if _count_matches(_RE_SWEDISH, text_norm) >= 1 and id_count <= 1:
            return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks terdeteksi bahasa Skandinavia (kata "nu")'
        if _count_matches(_RE_PORTUGUESE_SPANISH, text_norm) >= 2 and id_count <= 1:
            return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks terdeteksi bahasa Portugis/Spanyol'
        if _RE_MALAY_GOV.search(text_norm) and not _RE_NU_ISLAM_CONTEXT.search(text_norm):
            return 'OUT_OF_CONTEXT_LUAR_NEGERI', 'Teks merujuk program/kerajaan Malaysia'

    # 4. Dominasi bahasa Inggris tanpa entitas LII Indonesia
    en_count = _count_matches(_RE_EN_STOPWORDS, text_norm)
    if en_count >= 4 and id_count <= 1 and not has_any_tier_a:
        return 'OUT_OF_CONTEXT_BAHASA_ASING', 'Teks berbahasa Inggris tanpa entitas eksplisit LII'

    # 5. Homonim NU / Muhammadiyah / Spam
    if _RE_NU_HOMONYMS.search(text_norm) and not _RE_NU_TIER_A.search(text_norm):
        return 'OUT_OF_CONTEXT_HOMONIM_SPAM', 'Kecocokan homonim non-NU (game NU Carnival / universitas luar / mall NU Sentral / genre musik)'

    if topic_hint == 'Muhammadiyah' and _RE_AISYAH_PERSON.search(text_norm) and not _RE_MUH_TIER_A.search(text_norm):
        return 'OUT_OF_CONTEXT_HOMONIM_SPAM', 'Pencarian fuzzy "Aisyiyah" nyasar ke nama "Aisyah"'

    if _RE_OTHER_HOMONYMS.search(text_norm) and not has_any_tier_a:
        return 'OUT_OF_CONTEXT_HOMONIM_SPAM', 'Kecocokan homonim asing (IMM/AUM non-Muhammadiyah)'

    # 6. Sunda "nu" ("yang") — hanya bila TIDAK ada entitas Tier-A LII apa pun maupun konteks keislaman NU
    if topic_hint == 'Nahdlatul_Ulama' and not has_any_tier_a:
        alpha_chars = [ch for ch in text_norm if ch.isalpha()]
        is_all_caps = len(alpha_chars) > 12 and text_norm == text_norm.upper()
        has_upper_nu_civic = bool(_RE_BARE_NU_UPPER.search(text_norm)) and not is_all_caps and bool(_RE_NU_ISLAM_CONTEXT.search(text_norm))
        sunda_hits = _count_matches(_RE_SUNDA_MARKERS, text_norm)
        if not has_upper_nu_civic and _RE_BARE_NU_ANY.search(text_norm) and (sunda_hits >= 2 or _RE_SUNDA_NU_GRAMMAR.search(text_norm)):
            return 'OUT_OF_CONTEXT_SUNDA_BUKAN_NU', 'Kata "nu" merupakan bahasa Sunda ("yang"), bukan Nahdlatul Ulama'

    # 7. Spam iklan / judi / jualan / boilerplate (kecuali memang fatwa/sikap LII resmi)
    if _RE_SPAM_ADS.search(text_norm) and not (
        _RE_NU_TIER_A.search(text_norm) or _RE_MUH_TIER_A.search(text_norm) or _RE_BAZNAS_TIER_A.search(text_norm)
    ):
        return 'OUT_OF_CONTEXT_HOMONIM_SPAM', 'Iklan jualan, judol/slot, atau pesan sistem platform'

    # 8. Konten murni hashtag generik (#fyp #viral) tanpa isi teks
    plain_words = _strip_hashtags_mentions_urls(text_norm).split()
    if len(plain_words) < 3 and not has_any_tier_a and not _RE_NU_ISLAM_CONTEXT.search(text_norm):
        return 'OUT_OF_CONTEXT_HOMONIM_SPAM', 'Konten terlalu pendek / hanya berisi hashtag generik'

    return None, None


def _has_nu_context(text_norm):
    if _RE_NU_TIER_A.search(text_norm):
        return True
    if not _RE_BARE_NU_ANY.search(text_norm):
        return False

    alpha_chars = [ch for ch in text_norm if ch.isalpha()]
    is_all_caps = len(alpha_chars) > 12 and text_norm == text_norm.upper()
    has_upper_nu = bool(_RE_BARE_NU_UPPER.search(text_norm)) and not is_all_caps
    has_islam_civic = bool(_RE_NU_ISLAM_CONTEXT.search(text_norm))
    id_count = _count_matches(_RE_ID_STOPWORDS, text_norm)

    if has_upper_nu and (has_islam_civic or id_count >= 2):
        return True
    if _RE_BARE_NU_ANY.search(text_norm) and has_islam_civic and id_count >= 1:
        return True
    return False


def _has_muhammadiyah_context(text_norm):
    if _RE_MUH_TIER_A.search(text_norm):
        return True
    if _RE_MUH_SHORT_ORGS.search(text_norm) and _RE_MUH_CONTEXT.search(text_norm):
        return True
    return False


def _has_baznas_context(text_norm):
    if _RE_BAZNAS_TIER_A.search(text_norm):
        return True
    if _RE_UPZ.search(text_norm) and (_RE_ZAKAT_CORE.search(text_norm) or _RE_ID_STOPWORDS.search(text_norm)):
        return True
    # Konteks pengelolaan zakat/mustahik/muzakki hanya lolos ke BAZNAS jika tidak menyebut LAZ lain
    if _RE_ZAKAT_CORE.search(text_norm) and not _RE_NON_BAZNAS_ZAKAT_ORG.search(text_norm):
        if re.search(r'\b(?:muzakki|muzaki|mustahik|mustahiq|8\s+asnaf|delapan\s+asnaf|zakat\s+(?:fitrah|mal|maal|profesi|penghasilan|produktif|nasional)|pengelolaan\s+zakat|penyaluran\s+zakat)\b', text_norm, re.I):
            return True
    return False


_TOPIC_VALIDATORS = {
    'Nahdlatul_Ulama': _has_nu_context,
    'Muhammadiyah': _has_muhammadiyah_context,
    'BAZNAS': _has_baznas_context,
}


def classify_lii_relevance(raw_text, topic_hint, keyword='', platform=''):
    """
    Mengevaluasi relevansi 1 postingan terhadap 3 topik LII.

    Returns:
        tuple: (is_relevant: bool, reason_code: str, detail: str, resolved_topic: str | None)
    """
    text_norm = unicodedata.normalize('NFKD', str(raw_text or '')).strip()
    text_norm = re.sub(r'\s+', ' ', text_norm)
    if not text_norm:
        return False, 'TEKS_KOSONG', 'Konten teks kosong', None

    # 1. Deteksi bahasa asing, Sunda 'nu', homonim non-LII, dan spam
    noise_code, noise_detail = _detect_foreign_or_noise(text_norm, topic_hint=topic_hint)
    if noise_code:
        return False, noise_code, noise_detail, None

    # 2. Cek validator pada topik asal baris
    validator = _TOPIC_VALIDATORS.get(topic_hint)
    if validator and validator(text_norm):
        return True, 'RELEVAN', '', topic_hint

    # 3. Rekonsiliasi lintas-topik: misalnya "Selamat Milad Muhammadiyah" masuk lewat #nahdlatululama,
    #    atau "Lazismu Kedungbanteng" masuk lewat keyword BAZNAS.
    matched_tier_a_topics = [
        t_name for t_name, pat in _RE_RIVAL_TOPICS.items()
        if pat.search(text_norm)
    ]
    if len(matched_tier_a_topics) == 1:
        reconciled = matched_tier_a_topics[0]
        return True, 'RELEVAN_REKONSILIASI_TOPIK', f'Topik teks eksplisit merujuk ke {reconciled}', reconciled

    # Tangkap kasus Lazismu / Lazisnu khusus
    if re.search(r'\blazismu\b', text_norm, re.I) and not _RE_BAZNAS_TIER_A.search(text_norm):
        return True, 'RELEVAN_REKONSILIASI_TOPIK', 'Teks eksplisit membahas LAZISMU (Muhammadiyah)', 'Muhammadiyah'
    if re.search(r'\blazisnu\b', text_norm, re.I) and not _RE_BAZNAS_TIER_A.search(text_norm):
        return True, 'RELEVAN_REKONSILIASI_TOPIK', 'Teks eksplisit membahas LAZISNU (Nahdlatul Ulama)', 'Nahdlatul_Ulama'

    # 4. Tidak ada jangkar entitas maupun konteks LII di teks
    if topic_hint == 'Nahdlatul_Ulama':
        detail = 'Tidak memuat entitas Nahdlatul Ulama / NU atau konteks ormas Islam Indonesia'
    elif topic_hint == 'Muhammadiyah':
        detail = 'Tidak memuat entitas Muhammadiyah / Aisyiyah atau konteks persyarikatan yang valid'
    else:
        detail = 'Tidak memuat entitas BAZNAS / pengelolaan zakat nasional'

    return False, 'OUT_OF_CONTEXT_TIDAK_RELEVAN', detail, None
