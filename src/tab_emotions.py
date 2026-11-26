# pyrefly: ignore[missing-import]
import streamlit as st
import pandas as pd
from src.components import (
    plot_emotion_distribution,
    plot_emotion_by_platform,
    plot_emoji_analysis,
)

def render_tab_emotions(df_filtered):
    st.subheader("🧠 Analisis Emosi & Reaksi Simbolik (Emoji) Publik")
    st.caption("Pemetaan psikometrik persepsi massa menggunakan leksikon emosi Bahasa Indonesia dan simbol emotikon.")

    emo_col1, emo_col2 = st.columns([1, 1.4])
    with emo_col1:
        st.markdown("##### 🎭 Distribusi Emosi Publik")
        st.plotly_chart(plot_emotion_distribution(df_filtered), use_container_width=True, key="chart_emo_dist")
        st.caption("Klasifikasi 9 spektrum: Positif (Percaya, Antisipasi, Joy), Negatif (Jijik, Takut, Sedih, Marah), Netral (Surprise, Informatif).")

    with emo_col2:
        st.markdown("##### 📊 Komposisi Emosi per Platform Media Sosial")
        st.plotly_chart(plot_emotion_by_platform(df_filtered), use_container_width=True, key="chart_emo_platform")

    st.markdown("---")
    st.subheader("😃 Emoji Analysis: Simbol Ekspresi Publik Paling Populer")
    
    emoji_fig, emoji_counts = plot_emoji_analysis(df_filtered)
    if emoji_fig:
        st.plotly_chart(emoji_fig, use_container_width=True, key="chart_emoji_analysis")
        
        st.markdown("##### 💡 Wawasan Komunikasi Publik & Keumatan (LII):")
        st.markdown("""
        * **Emoji Bernada Dukungan (👍, ❤️, 🙏, 🔥, 👏, 🇮🇩)**: Merefleksikan apresiasi masyarakat akar rumput terhadap khidmat pendidikan pesantren/sekolah, layanan kesehatan, tanggap bencana, serta penyaluran zakat dan beasiswa umat (NU, Muhammadiyah, & BAZNAS).
        * **Emoji Bernada Skeptis / Emosi Negatif (😡, 💀, 😞, ⚠️)**: Umumnya berkorelasi dengan sorotan kritis terhadap dinamika elite organisasi, pro-kontra isu konsesi/politik praktis, atau tuntutan transparansi akuntabilitas pengelolaan dana umat.
        * **Rekomendasi Komunikasi Strategis LII**: Memperbanyak publikasi laporan dampak sosial-kemanusiaan yang transparan dan narasi kesejukan lintas ormas untuk mengonversi emosi cemas/marah (*fear/anger*) menjadi kepercayaan publik (*trust/joy*).
        """)
    else:
        st.info("Tidak ditemukan emoji pada subset data yang dipilih.")
