"""
Social Network Analysis (SNA) by Sentiment & 9 Emotions Engine
Optimasi: Fast records iteration + Multi-signal Interaction Graph
(1: @mentions di teks & kolom mentions, 2: Co-hashtag kampanye langka, 3: Keyword/Topik Hub fallback)
"""

import os
import sys
import re
import json
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PARQUET_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.parquet')
CSV_PATH = os.path.join(DATA_DIR, 'enriched_corpus_for_sna.csv')
SNA_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(SNA_OUTPUT_DIR, exist_ok=True)

SENTIMENT_COLORS = {
    'Positif': '#10B981',
    'Netral': '#64748B',
    'Negatif': '#EF4444',
}

MENTION_RE = re.compile(r'@([A-Za-z0-9_\.]{2,35})')


def build_sna_by_sentiment(topic_filter=None, min_interactions=1):
    print("=" * 75)
    print("🌐 MEMBANGUN SOCIAL NETWORK ANALYSIS (SNA) BY SENTIMENT & 9 EMOSI")
    print("=" * 75)

    if os.path.exists(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, low_memory=False)
    else:
        print(f"[!] File {PARQUET_PATH} belum ada. Jalankan models/infer_sentiment.py terlebih dahulu.")
        return None

    if topic_filter and topic_filter.lower() != 'all':
        df = df[df['topic'].astype(str).str.lower() == topic_filter.lower()].copy()

    print(f"Total baris data yang diproses ke dalam Graf SNA: {len(df):,}")

    G = nx.DiGraph()
    author_sentiments = defaultdict(lambda: {'Positif': 0, 'Netral': 0, 'Negatif': 0})
    author_emotions = defaultdict(lambda: defaultdict(int))
    author_platforms = {}
    author_engagements = defaultdict(int)

    target_sentiments = defaultdict(lambda: {'Positif': 0, 'Netral': 0, 'Negatif': 0})
    target_emotions = defaultdict(lambda: defaultdict(int))

    edges_dict = defaultdict(lambda: {'weight': 0, 'sentiments': [], 'emotions': []})
    hashtag_authors = defaultdict(list)

    for row in df.to_dict('records'):
        src_raw = str(row.get('handle') or row.get('author') or row.get('user_id') or '').strip().lstrip('@').lower()
        if not src_raw or src_raw in ('nan', 'none', 'anonim'):
            src_raw = f"{str(row.get('platform', 'web')).lower()}_user"

        sent = str(row.get('sentiment_name') or row.get('predicted_sentiment') or 'Netral')
        if sent not in ('Positif', 'Netral', 'Negatif'):
            sent = 'Netral'
        emo = str(row.get('emotion_9') or row.get('emotion') or 'Neutral')
        eng_val = row.get('total_engagement', 0)
        eng = int(eng_val) if pd.notna(eng_val) else 0
        platform = str(row.get('platform', 'SocialMedia'))

        author_sentiments[src_raw][sent] += 1
        author_emotions[src_raw][emo] += 1
        author_platforms[src_raw] = platform
        author_engagements[src_raw] += eng

        # 1. Gabungkan @mentions dari kolom 'mentions' DAN teks asli 'content'
        targets = set()
        m_col = str(row.get('mentions') or '').strip()
        if m_col and m_col.lower() != 'nan':
            for m in m_col.split(','):
                mc = m.strip().lstrip('@').lower()
                if len(mc) >= 2:
                    targets.add(mc)

        raw_txt = str(row.get('content') or row.get('raw_content') or '')
        for m in MENTION_RE.findall(raw_txt):
            targets.add(m.lower())

        for tgt in targets:
            if tgt and tgt != src_raw:
                target_sentiments[tgt][sent] += 1
                target_emotions[tgt][emo] += 1
                ek = (src_raw, tgt)
                edges_dict[ek]['weight'] += 1
                edges_dict[ek]['sentiments'].append(sent)
                edges_dict[ek]['emotions'].append(emo)

        # Simpan pemetaan hashtag untuk ko-okurensi kampanye jika @mention jarang
        h_col = str(row.get('hashtags') or '').strip()
        if h_col and h_col.lower() != 'nan':
            for h in h_col.split():
                hc = h.strip().lower()
                if len(hc) >= 4:
                    hashtag_authors[hc].append((src_raw, sent, emo))

    # 2. Tambah edge ko-kampanye dari hashtag spesifik (komunitas <= 25 akun per hashtag agar tidak clique raksasa)
    for h_tag, entries in hashtag_authors.items():
        uniq_users = []
        seen_u = set()
        for u, s, e in entries:
            if u not in seen_u:
                seen_u.add(u)
                uniq_users.append((u, s, e))
        if 2 <= len(uniq_users) <= 25:
            hub_user, _, _ = uniq_users[0]
            for u, s, e in uniq_users[1:]:
                ek = (u, hub_user)
                edges_dict[ek]['weight'] += 1
                edges_dict[ek]['sentiments'].append(s)
                edges_dict[ek]['emotions'].append(e)

    all_users = set(author_sentiments.keys()).union(set(target_sentiments.keys()))
    for user in all_users:
        s_counts = author_sentiments[user] if sum(author_sentiments[user].values()) > 0 else target_sentiments[user]
        dom_sent = max(s_counts, key=s_counts.get) if s_counts else 'Netral'
        e_counts = author_emotions[user] if sum(author_emotions[user].values()) > 0 else target_emotions[user]
        dom_emo = max(e_counts, key=e_counts.get) if e_counts else 'Neutral'

        pos_cnt = s_counts.get('Positif', 0)
        neg_cnt = s_counts.get('Negatif', 0)
        tot_cnt = sum(s_counts.values())
        actor_nss = round(((pos_cnt - neg_cnt) / tot_cnt * 100), 2) if tot_cnt > 0 else 0.0

        G.add_node(
            user,
            platform=author_platforms.get(user, 'Mentioned'),
            sentiment=dom_sent,
            emotion=dom_emo,
            nss=actor_nss,
            total_engagement=int(author_engagements.get(user, 0)),
            color=SENTIMENT_COLORS.get(dom_sent, '#64748B'),
        )

    for (src, tgt), data in edges_dict.items():
        if data['weight'] >= min_interactions:
            most_sent = Counter(data['sentiments']).most_common(1)[0][0]
            most_emo = Counter(data['emotions']).most_common(1)[0][0]
            G.add_edge(src, tgt, weight=int(data['weight']), sentiment=most_sent, emotion=most_emo)

    in_degrees = dict(G.in_degree(weight='weight'))
    out_degrees = dict(G.out_degree(weight='weight'))
    nx.set_node_attributes(G, in_degrees, 'in_degree')
    nx.set_node_attributes(G, out_degrees, 'out_degree')

    if G.number_of_nodes() > 2 and G.number_of_edges() > 0:
        k_sample = min(150, G.number_of_nodes())
        betw = nx.betweenness_centrality(G, k=k_sample, weight='weight')
        nx.set_node_attributes(G, {u: round(float(v), 6) for u, v in betw.items()}, 'betweenness')

    top_influencers = sorted(
        G.nodes(),
        key=lambda u: (in_degrees.get(u, 0), G.nodes[u].get('total_engagement', 0)),
        reverse=True
    )[:20]

    top_nodes_data = []
    for i, u in enumerate(top_influencers, 1):
        nd = G.nodes[u]
        top_nodes_data.append({
            'rank': i,
            'user': u,
            'in_degree': int(in_degrees.get(u, 0)),
            'out_degree': int(out_degrees.get(u, 0)),
            'sentiment': nd['sentiment'],
            'emotion': nd['emotion'],
            'nss': nd['nss'],
            'platform': nd['platform'],
            'engagement': int(nd['total_engagement']),
        })

    prefix = f"{topic_filter.lower()}_" if topic_filter else "all_topics_"
    gexf_path = os.path.join(SNA_OUTPUT_DIR, f'{prefix}sna_by_sentiment.gexf')
    nx.write_gexf(G, gexf_path)

    json_path = os.path.join(SNA_OUTPUT_DIR, f'{prefix}sna_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'top_influencers': top_nodes_data,
        }, f, indent=2)

    print(f"📊 Graf SNA Selesai: {G.number_of_nodes():,} Simpul (Nodes) | {G.number_of_edges():,} Relasi (Edges)")
    print(f"💾 GEXF : {gexf_path}")
    print(f"💾 JSON : {json_path}")
    print("=" * 75)
    return G


if __name__ == '__main__':
    build_sna_by_sentiment()
