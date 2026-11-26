# 🛠️ Dokumentasi Tools & Automation Project

Folder ini berisi kumpulan utilitas independen untuk pengelolaan kredensial, sampling gold standard, anotasi LLM (Gemini), audit model IndoBERT, serta otomasi notebook Google Colab.

---

## 📂 Daftar Utilitas & Fungsinya

### 1. Evaluasi & Audit Model IndoBERT
* **[`audit_model.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/audit_model.py)**:
  * Melakukan spot-check acak pada prediksi sentimen (Positif, Negatif, Netral), korelasi emosi Plutchik, adversarial check (kata-kata anomali), serta inventaris dataset.
  * Cara eksekusi: `python tools/audit_model.py`
* **[`audit_deep.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/audit_deep.py)**:
  * Melakukan analisis adversarial mendalam pada postingan paradoks (misal kata *"keracunan"* tapi skor positif, atau kata *"berhasil"* tapi skor negatif).
  * Cara eksekusi: `python tools/audit_deep.py`

### 2. Anotasi Gold Standard (Sampling & LLM Labeling)
* **[`step1_sample_gold_standard.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/step1_sample_gold_standard.py)**:
  * Mengambil stratified sample 1.000 data seimbang lintas 7 platform dan 3 topik kebijakan untuk anotasi standar emas.
  * Output: `data/gold_standard_sample_1000.csv`
* **[`step2_gemini_annotation.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/step2_gemini_annotation.py)**:
  * Pipeline anotasi otomatis menggunakan Google Gemini API (dengan multi-model fallback, batch processing, dan checkpoint resume).
  * Output: `data/gold_standard_annotated_1000.csv`
* **[`test_gemini_9emotions.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/test_gemini_9emotions.py)**:
  * Pengujian cepat klasifikasi 9 kategori emosi Plutchik via Gemini API.
* **[`samples_9_emotions.json`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/samples_9_emotions.json)**:
  * Sampel rujukan untuk 9 kategori emosi (*Joy, Sadness, Anger, Fear, Trust, Disgust, Surprise, Anticipation, Neutral*).

### 3. Pipeline Colab & Pelatihan Ulang GPU
* **[`colab_full_pipeline_v2.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/colab_full_pipeline_v2.py)**:
  * Script lengkap all-in-one untuk Google Colab (pelatihan ulang IndoBERT Sentiment & Emotion di GPU T4).
* **[`make_colab_notebook.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/make_colab_notebook.py)**:
  * Mengonversi `colab_full_pipeline_v2.py` menjadi notebook `.ipynb` yang siap di-upload ke Colab.

### 4. Manajemen Akun, Cookie & Kredensial
* **[`.env`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/.env)**:
  * Menyimpan seluruh API key (YouTube Data API, Twitter cookies, TikTok cookies, Apify, dll).
* **[`accounts.db`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/accounts.db)**:
  * Database SQLite untuk session pool akun Twitter (twscrape).
* **[`add_cookies.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/add_cookies.py)**:
  * Mempermudah penambahan template cookie Twitter ke `.env`.
* **[`add_env.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/add_env.py)**:
  * Mempermudah penambahan placeholder akun dummy twscrape ke `.env`.
* **[`get_token.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/get_token.py)**:
  * Pengambilan access token Facebook Graph / Instagram Graph API.
* **[`reset_pool.py`](file:///d:/Project_Streamlit/IndoBERT_SNA_MT_PP/tools/reset_pool.py)**:
  * Menghapus cache state akun Twitter di `accounts.db` jika terkena rate-limit/ban.
