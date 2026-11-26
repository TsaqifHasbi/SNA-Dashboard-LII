# IndoBERT & Multiplatform SNA: Riset Opini Publik Elektrifikasi Kendaraan di Indonesia

**Judul Penelitian Tugas Akhir**:  
*Integrasi IndoBERT dan Multiplatform Social Network Analysis sebagai Sistem Marketing Intelligence pada Riset Opini Publik Elektrifikasi Kendaraan di Indonesia*  
**Program Studi**: S1 Teknik Informatika, Fakultas Teknik, Universitas Jenderal Soedirman (UNSOED)  
**Dosen Pembimbing**: Muhammad Ihsan Zul, S.Kom., M.Eng.  

---

## 📌 Ringkasan Eksekutif Proyek

Repositori ini merupakan implementasi komputasional end-to-end untuk riset opini publik multiplatform mengenai **Elektrifikasi Kendaraan di Indonesia**. Sistem mengintegrasikan:
1. **Multi-Source Scraping (7 Kanal)**: Twitter/X, Threads, Instagram, Facebook, TikTok, YouTube, dan Portal Berita Daring.
2. **Deep Learning NLP (IndoBERT)**:
   - Klasifikasi Polaritas Sentimen 3-Kelas (*Positif*, *Netral*, *Negatif*).
   - Klasifikasi Dimensi Emosi 9-Kelas Robert Plutchik (*Joy*, *Sadness*, *Anger*, *Fear*, *Trust*, *Disgust*, *Surprise*, *Anticipation*, *Neutral*).
3. **Multiplatform Social Network Analysis (SNA)**:
   - Pemetaan relasi aktor, identifikasi *Key Opinion Leader* (KOL) via *Degree*, *Betweenness*, *Closeness*, dan *Eigenvector Centrality*.
   - Deteksi komunitas modularitas Louvain.
   - Proyeksi jaringan berbasis sentimen (*SNA by Sentiment*) dan emosi (*SNA by Emotion*).
4. **Sistem Rekomendasi Solusi & Dashboard Marketing Intelligence**:
   - Analisis framing Entman & Edelman.
   - Evaluasi kebermaknaan pergeseran sentimen berbasis *Two-Sample Z-Test*.
   - Visualisasi graf interaktif PyVis dan ekspor Gephi (`.gexf`).

---

## 🗂️ Struktur Direktori Proyek

```text
IndoBERT_SNA_MT_EV/
├── colab_cells/                   # Mesin Scraping 7 Platform & Konfigurasi
│   ├── cell_02_config_elektrifikasi.py # 45 Keyword, Regex, & Direktori TA
│   ├── cell_03_youtube.py         # Engine YouTube Data API v3
│   ├── cell_04_twitter.py         # Engine Twitter/X Scraper (Selenium/API)
│   ├── cell_05_tiktok.py          # Engine TikTok Scraper
│   ├── cell_06_news.py            # Engine Portal Berita Online Scraper
│   ├── cell_07_instagram.py       # Engine Instagram Scraper (Full Captions)
│   ├── cell_08_threads.py         # Engine Meta Threads Scraper
│   ├── cell_09_facebook.py        # Engine Meta Facebook Scraper
│   ├── run_ta_scraper.py          # Master CLI Scraper Runner
│   ├── run_ta_instagram.py        # Runner Mandiri Instagram
│   ├── run_ta_facebook.py         # Runner Mandiri Facebook
│   ├── run_ta_twitter.py          # Runner Mandiri Twitter
│   ├── run_ta_threads.py          # Runner Mandiri Threads
│   ├── run_ta_tiktok.py           # Runner Mandiri TikTok
│   ├── run_ta_youtube.py          # Runner Mandiri YouTube
│   └── run_ta_news.py             # Runner Mandiri Berita Daring
├── ScrapeData/   # Korpus Data Mentah & Parsial 7 Platform (EV)
│   ├── scraped_Facebook_Only.csv
│   ├── scraped_Instagram_Only.csv
│   ├── scraped_News_Only.csv
│   ├── scraped_Threads_Only.csv
│   ├── scraped_TikTok_Only.csv
│   ├── scraped_Twitter_Only.csv
│   └── scraped_YouTube_Only.csv
├── ScrapeData/                    # Junction Direktori ke ScrapeData
├── data/                          # Dataset Siap Training & Enriched Corpus
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── enriched_corpus_for_sna.parquet
├── models/                        # Script Fine-Tuning & Intelejen IndoBERT
│   ├── dataset_preparation.py     # Ekstraksi korpus & stratifikasi data latih/uji
│   ├── prepare_9_emotions_dataset.py # Pemetaan dataset seimbang 9 emosi
│   ├── train_indobert.py          # Fine-tuning IndoBERT Sentimen 3-Kelas
│   ├── train_indobert_9_emotions.py # Fine-tuning IndoBERT 9-Emosi
│   ├── infer_sentiment.py         # Batch inference korpus multiplatform
│   └── best_indobert_sentiment/   # Model checkpoint sentimen
│   └── best_indobert_emotion/     # Model checkpoint emosi
├── src/                           # Backend Dashboard, Kalkulator Metrik, & SNA
│   ├── sna_engine.py              # Algoritma SNA NetworkX, PyVis, Gephi
│   ├── metrics_calculator.py      # NSS, Z-Test, Framing Entman
│   ├── data_loader.py             # Caching data Parquet / CSV
│   └── components.py              # Visualisasi Plotly & Dual Word Cloud
├── tools/                         # Utilitas Anotasi & Sampling Gemini
│   ├── step1_sample_gold_standard.py # Sampling stratified multiplatform
│   └── step2_gemini_annotation.py    # Auto-labeling Gold Standard via LLM
├── sna/output/                    # File Graf Jaringan (.gexf) & Tabel Metrik
├── accounts.db                    # Database Lokal Kredensial & Session Cookie
├── IndoBERT_SNA_Training_Pipeline.ipynb # Jupyter/Colab Pipeline Training
└── requirements.txt               # Daftar Pustaka Python
```

---

## 🚀 Panduan Menjalankan Scraping Data TA

### 1. Menjalankan Scraping Kanal Spesifik
Setiap kanal dapat dijalankan secara independen di terminal:

```powershell
# Instagram (Auto-expand caption & anti-truncation)
python colab_cells/run_ta_instagram.py

# Twitter / X
python colab_cells/run_ta_twitter.py

# Facebook
python colab_cells/run_ta_facebook.py

# Threads
python colab_cells/run_ta_threads.py

# TikTok
python colab_cells/run_ta_tiktok.py

# YouTube
python colab_cells/run_ta_youtube.py

# Berita Daring
python colab_cells/run_ta_news.py
```

### 2. Menjalankan Master Scraper (Semua Kanal Berurutan)
```powershell
python colab_cells/run_ta_scraper.py --platform all
```

---

## 📊 Matriks 45 Variasi Kata Kunci (3 Pilar Topik)

1. **Elektrifikasi Kendaraan (Ekosistem, SPKLU, Kebijakan)**:
   - *elektrifikasi kendaraan, kendaraan listrik, kelebihan kendaraan listrik, manfaat elektrifikasi kendaraan, kendaraan listrik ramah lingkungan, subsidi kendaraan listrik, SPKLU, SPKLU PLN, cas kendaraan listrik, stasiun pengisian kendaraan listrik, kelemahan kendaraan listrik, masalah kendaraan listrik, SPKLU rusak, antre SPKLU, limbah baterai kendaraan listrik*.
2. **Mobil Listrik (EV Roda 4, PPN DTP, Adopsi Pasar)**:
   - *mobil listrik, kelebihan mobil listrik, keuntungan mobil listrik, puas pakai mobil listrik, kelebihan baterai mobil listrik, insentif mobil listrik, PPN DTP mobil listrik, subsidi mobil listrik, ngecas mobil listrik, home charging mobil listrik, kekurangan mobil listrik, masalah mobil listrik, baterai mobil listrik rusak, mobil listrik mogok, harga baterai mobil listrik mahal*.
3. **Motor Listrik (EV Roda 2, Subsidi, Konversi)**:
   - *motor listrik, kelebihan motor listrik, keuntungan motor listrik, puas pakai motor listrik, kelebihan baterai motor listrik, subsidi motor listrik, program konversi motor listrik, swap baterai motor listrik, hemat motor listrik, ngecas motor listrik, kekurangan motor listrik, masalah motor listrik, motor listrik mogok, baterai motor listrik cepat habis, tarik motor listrik lambat*.

---

## 🔬 Tahap Modeling & Training IndoBERT
Setelah korpus data terkumpul:
1. Siapkan dataset seimbang:
   ```powershell
   python models/dataset_preparation.py
   python models/prepare_9_emotions_dataset.py
   ```
2. Fine-tuning model IndoBERT pada Colab GPU / GPU Lokal menggunakan notebook `IndoBERT_SNA_Training_Pipeline.ipynb` atau script `python models/train_indobert.py`.

