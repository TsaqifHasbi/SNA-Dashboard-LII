"""
Langkah 2: Anotasi Gold Standard dengan Gemini API (3 Sentimen + 9 Emosi Plutchik)
Mendukung multi-model fallback resmi + checkpoint resume otomatis.
"""
import os
import sys
import json
import time
import re
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
COLAB_DIR = os.path.join(PROJECT_ROOT, 'colab_cells')
if COLAB_DIR not in sys.path:
    sys.path.insert(0, COLAB_DIR)

try:
    from cell_02_config import PROJECT_TITLE
except Exception:
    PROJECT_TITLE = "Opini Publik Indonesia"

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except Exception:
    pass

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INPUT_PATH = os.path.join(DATA_DIR, 'gold_standard_sample_1000.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'gold_standard_annotated_1000.csv')
CHECKPOINT_PATH = os.path.join(DATA_DIR, 'annotation_checkpoint.json')

BATCH_SIZE = 5
MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-lite']

VALID_SENTIMENTS = {'Positif', 'Netral', 'Negatif'}
VALID_EMOTIONS_9 = {'Trust', 'Anticipation', 'Joy', 'Disgust', 'Fear', 'Sadness', 'Anger', 'Surprise', 'Neutral'}
EMO_LABEL_MAP = {name: idx for idx, name in enumerate(['Trust', 'Anticipation', 'Joy', 'Disgust', 'Fear', 'Sadness', 'Anger', 'Surprise', 'Neutral'])}

PROMPT = """Kamu ahli linguistik & analisis sentimen Bahasa Indonesia untuk domain {domain}.

Berikan label untuk setiap teks:
- sentiment: "Positif" / "Netral" / "Negatif"
- emotion: pilih TEPAT SATU dari 9 emosi Plutchik berikut:
  * Positif : "Trust" / "Anticipation" / "Joy"
  * Negatif : "Disgust" / "Fear" / "Sadness" / "Anger"
  * Netral  : "Surprise" / "Neutral"
- confidence: 0.0-1.0
- reason: alasan singkat 1 kalimat

PENTING: Pahami konteks utuh, sarkasme, dan negasi (bukan kata per kata). Berita objektif informatif = "Netral" & "Neutral".
Output HANYA JSON array valid tanpa markdown:
[{{"idx":0,"sentiment":"Netral","emotion":"Neutral","confidence":0.9,"reason":"..."}}]

TEKS:
{texts}"""


def _parse_json_response(text, expected_n):
    t = (text or '').strip()
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z]*\n?', '', t)
        t = re.sub(r'\n?```$', '', t).strip()
    results = None
    try:
        results = json.loads(t)
    except Exception:
        m = re.search(r'\[.*\]', t, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group())
            except Exception:
                pass
    if not isinstance(results, list):
        return None
    clean = []
    for item in results:
        if isinstance(item, dict) and 'sentiment' in item:
            clean.append(item)
        else:
            clean.append({"idx": len(clean), "sentiment": "Netral", "emotion": "Neutral", "confidence": 0.5, "reason": "default"})
    while len(clean) < expected_n:
        clean.append({"idx": len(clean), "sentiment": "Netral", "emotion": "Neutral", "confidence": 0.3, "reason": "padding"})
    return clean[:expected_n]


def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set environment variable GEMINI_API_KEY terlebih dahulu sebelum menjalankan anotasi Gemini.")

    # pyrefly: ignore [missing-import]
    from google import genai
    client = genai.Client(api_key=api_key)

    print("=" * 70)
    print(f"🤖 ANOTASI GOLD STANDARD GEMINI API — {PROJECT_TITLE}")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)
    cp = json.load(open(CHECKPOINT_PATH, encoding='utf-8')) if os.path.exists(CHECKPOINT_PATH) else {"done": [], "ann": []}
    done = set(cp.get("done", []))
    anns = cp.get("ann", [])

    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    t0 = time.time()

    for bn, bdf in enumerate(batches):
        if bn in done:
            continue
        texts_block = "".join(
            f'\n[{i}] ({row.get("platform", "")}/{row.get("topic", "")}) "{str(row["cleaned_content"])[:420]}"\n'
            for i, (_, row) in enumerate(bdf.iterrows())
        )
        prompt_str = PROMPT.format(domain=PROJECT_TITLE, texts=texts_block)
        batch_res = None
        for model_name in MODELS:
            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt_str,
                        config={'temperature': 0.1, 'max_output_tokens': 2048},
                    )
                    batch_res = _parse_json_response(resp.text, len(bdf))
                    if batch_res:
                        break
                except Exception as e:
                    if '429' in str(e) or '503' in str(e):
                        time.sleep(4 * (attempt + 1))
                    else:
                        break
            if batch_res:
                break

        if not batch_res:
            batch_res = [{"idx": i, "sentiment": "Netral", "emotion": "Neutral", "confidence": 0.0, "reason": "fallback"} for i in range(len(bdf))]

        for i, (di, row) in enumerate(bdf.iterrows()):
            r = batch_res[i]
            s_val = r.get('sentiment', 'Netral')
            e_val = r.get('emotion', 'Neutral')
            s_val = s_val if s_val in VALID_SENTIMENTS else 'Netral'
            e_val = e_val if e_val in VALID_EMOTIONS_9 else 'Neutral'
            anns.append({
                "original_index": int(di),
                "platform": row.get('platform', ''),
                "topic": row.get('topic', ''),
                "cleaned_content": str(row['cleaned_content'])[:500],
                "old_sentiment": str(row.get('sentiment_name', '')),
                "old_emotion": str(row.get('emotion', '')),
                "gemini_sentiment": s_val,
                "gemini_emotion": e_val,
                "gemini_confidence": float(r.get('confidence', 0.5)),
                "gemini_reason": str(r.get('reason', '')),
            })

        done.add(bn)
        if (bn + 1) % 10 == 0 or (bn + 1) == len(batches):
            with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                json.dump({"done": list(done), "ann": anns}, f, ensure_ascii=False)
        print(f"  Batch {bn+1}/{len(batches)} selesai ({len(done)/len(batches)*100:.0f}%) | {time.time()-t0:.0f}s")
        time.sleep(0.25)

    res_df = pd.DataFrame(anns)
    res_df['sentiment_label'] = res_df['gemini_sentiment'].map({'Negatif': 0, 'Netral': 1, 'Positif': 2})
    res_df['emotion_label'] = res_df['gemini_emotion'].map(EMO_LABEL_MAP)
    res_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
    print(f"\n✅ Selesai! {len(res_df):,} data Gold Standard tersimpan ke: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
