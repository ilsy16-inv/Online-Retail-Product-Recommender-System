import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Basket Analysis — Online Retail",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0D1117;
    --surface:   #161B22;
    --surface2:  #21262D;
    --border:    #30363D;
    --accent:    #58A6FF;
    --accent2:   #F78166;
    --accent3:   #3FB950;
    --text:      #E6EDF3;
    --muted:     #8B949E;
    --tag-bg:    #1F2933;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover { border-color: var(--accent); }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }
.stDataFrame { background: var(--surface) !important; }

/* Sliders */
[data-testid="stSlider"] { color: var(--accent); }
.stSlider > div > div > div { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0D1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Select / input */
.stSelectbox > div, .stTextInput > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border);
    gap: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-weight: 500;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background: var(--surface2) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Plotly charts bg */
.js-plotly-plot { background: var(--surface) !important; }

/* Custom cards */
.kpi-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.kpi-card h4 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    margin: 0 0 0.25rem;
    font-size: 1.1rem;
}
.kpi-card p { margin: 0; color: var(--muted); font-size: 0.85rem; }

.rule-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.rule-card:hover { border-left-color: var(--accent2); }
.rule-card .ant { color: var(--accent2); font-weight: 600; font-size: 0.9rem; }
.rule-card .con { color: var(--accent3); font-weight: 600; font-size: 0.9rem; }
.rule-card .metrics { display: flex; gap: 1.5rem; margin-top: 0.5rem; }
.rule-card .metric-item { font-size: 0.78rem; color: var(--muted); }
.rule-card .metric-val { font-family: 'Space Mono', monospace; color: var(--text); font-size: 0.85rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}

.badge {
    display: inline-block;
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    color: var(--muted);
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
}
.accent-dot { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161B22",
    plot_bgcolor="#161B22",
    font=dict(family="DM Sans", color="#E6EDF3"),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#58A6FF", "#F78166", "#3FB950", "#D2A8FF", "#FFA657", "#79C0FF"],
)


# ─── DATA LOADING & PROCESSING ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df_clean = df.dropna(subset=["CustomerID", "Description"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean["Description"] = df_clean["Description"].str.strip()
    return df, df_clean


@st.cache_data(show_spinner=False)
def build_rules(df_clean, country, min_support, min_confidence, max_len):
    df_country = df_clean[df_clean["Country"] == country] if country != "All" else df_clean
    basket = (
        df_country.groupby("InvoiceNo")["Description"]
        .apply(lambda items: list(set(items)))
        .reset_index(drop=True)
        .tolist()
    )
    te = TransactionEncoder()
    te_array = te.fit(basket).transform(basket)
    df_enc = pd.DataFrame(te_array, columns=te.columns_)
    freq = apriori(df_enc, min_support=min_support, use_colnames=True, max_len=max_len)
    freq["length"] = freq["itemsets"].apply(len)
    if freq.empty:
        return freq, pd.DataFrame(), basket
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return freq, rules, basket


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">📁 Dataset</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Online_Retail.xlsx", type=["xlsx"])

    st.markdown('<p class="section-title" style="margin-top:1.5rem">⚙️ Parameters</p>', unsafe_allow_html=True)
    min_sup = st.slider("Min Support", 0.005, 0.10, 0.015, 0.005, format="%.3f",
                        help="Minimum frekuensi kombinasi produk dari total transaksi")
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.30, 0.05, format="%.2f",
                         help="Minimum probabilitas pembelian consequent jika antecedent dibeli")
    max_len_val = st.slider("Max Itemset Length", 2, 4, 3,
                            help="Maksimum jumlah produk dalam satu kombinasi")

    country_sel = None
    df_raw = None
    df_clean = None

    if uploaded:
        with st.spinner("Loading data..."):
            df_raw, df_clean = load_and_clean(uploaded)
        countries = ["All"] + sorted(df_clean["Country"].unique().tolist())
        country_sel = st.selectbox("Filter Country", countries, index=countries.index("United Kingdom") if "United Kingdom" in countries else 0)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#8B949E;">Market Basket Analysis<br>Algoritma Apriori · Online Retail</p>', unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.5rem">
  <h1 class="hero-title">🛒 Market Basket <span class="accent-dot">Analysis</span></h1>
  <p class="hero-sub">Sistem Rekomendasi Produk · Algoritma Apriori · Online Retail Dataset</p>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.info("⬆️ Upload file **Online_Retail.xlsx** di sidebar untuk memulai analisis.")
    st.markdown("""
    <div class="kpi-card">
        <h4>Cara Penggunaan</h4>
        <p>1. Upload file <code>Online_Retail.xlsx</code> di sidebar kiri</p>
        <p style="margin-top:0.4rem">2. Atur parameter <b>Min Support</b>, <b>Min Confidence</b>, dan <b>Max Itemset Length</b></p>
        <p style="margin-top:0.4rem">3. Pilih negara target (default: United Kingdom)</p>
        <p style="margin-top:0.4rem">4. Eksplorasi insight di tab-tab yang tersedia</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Build rules ──
with st.spinner("⏳ Menjalankan Apriori..."):
    freq_items, rules_df, basket = build_rules(df_clean, country_sel, min_sup, min_conf, max_len_val)

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔗 Association Rules",
    "🔍 Rekomendasi Produk",
    "📈 Visualisasi",
    "📋 Data Explorer"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    df_country_view = df_clean[df_clean["Country"] == country_sel] if country_sel != "All" else df_clean

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Baris (clean)", f"{len(df_clean):,}")
    c2.metric("Transaksi", f"{df_country_view['InvoiceNo'].nunique():,}")
    c3.metric("Produk Unik", f"{df_country_view['Description'].nunique():,}")
    c4.metric("Frequent Itemsets", f"{len(freq_items):,}" if not freq_items.empty else "0")
    c5.metric("Rules Ditemukan", f"{len(rules_df):,}" if not rules_df.empty else "0")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-title">Top 15 Produk Terlaris</p>', unsafe_allow_html=True)
        top_p = (
            df_country_view.groupby("Description")["Quantity"]
            .sum().sort_values(ascending=False).head(15)
        )
        fig = px.bar(
            x=top_p.values, y=top_p.index,
            orientation="h",
            labels={"x": "Total Quantity", "y": ""},
            color=top_p.values,
            color_continuous_scale=["#1F2933", "#58A6FF"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          coloraxis_showscale=False, height=420,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-title">Distribusi Jumlah Item per Transaksi</p>', unsafe_allow_html=True)
        basket_sizes = df_country_view.groupby("InvoiceNo")["Description"].count().clip(upper=60)
        fig2 = px.histogram(
            basket_sizes, nbins=60,
            labels={"value": "Jumlah Item", "count": "Frekuensi"},
            color_discrete_sequence=["#58A6FF"]
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=420, bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)

    # Country distribution
    st.markdown('<p class="section-title">Distribusi Transaksi per Negara (Top 10)</p>', unsafe_allow_html=True)
    top_countries = df_clean["Country"].value_counts().head(10)
    fig3 = px.bar(
        x=top_countries.index, y=top_countries.values,
        labels={"x": "Negara", "y": "Jumlah Transaksi"},
        color=top_countries.values,
        color_continuous_scale=["#1F2933", "#F78166"],
    )
    fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, coloraxis_showscale=False, height=300)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if rules_df.empty:
        st.warning("⚠️ Tidak ada rules yang ditemukan. Coba turunkan nilai Min Support atau Min Confidence.")
    else:
        col_sort, col_top = st.columns([2, 1])
        sort_by = col_sort.selectbox("Urutkan berdasarkan", ["lift", "confidence", "support"], index=0)
        top_n = col_top.number_input("Tampilkan N rules teratas", 5, 50, 15, step=5)

        rules_show = rules_df.sort_values(sort_by, ascending=False).head(top_n)

        st.markdown(f'<p class="section-title">Top {top_n} Rules — diurutkan berdasarkan {sort_by.upper()}</p>', unsafe_allow_html=True)

        for _, row in rules_show.iterrows():
            ant = ", ".join(list(row["antecedents"]))
            con = ", ".join(list(row["consequents"]))
            st.markdown(f"""
            <div class="rule-card">
                <span class="ant">IF: {ant}</span>
                <span style="color:#8B949E;margin:0 0.5rem">→</span>
                <span class="con">THEN: {con}</span>
                <div class="metrics">
                    <div class="metric-item">Support<br><span class="metric-val">{row['support']:.4f} ({row['support']*100:.1f}%)</span></div>
                    <div class="metric-item">Confidence<br><span class="metric-val">{row['confidence']:.4f} ({row['confidence']*100:.1f}%)</span></div>
                    <div class="metric-item">Lift<br><span class="metric-val" style="color:#3FB950">{row['lift']:.2f}x</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Itemset length distribution
        if not freq_items.empty:
            st.markdown('<p class="section-title">Distribusi Panjang Frequent Itemsets</p>', unsafe_allow_html=True)
            len_dist = freq_items["length"].value_counts().sort_index()
            fig_len = px.bar(
                x=[f"{i} item" for i in len_dist.index], y=len_dist.values,
                labels={"x": "", "y": "Jumlah Itemset"},
                color_discrete_sequence=["#58A6FF"]
            )
            fig_len.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig_len, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REKOMENDASI PRODUK
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if rules_df.empty:
        st.warning("⚠️ Tidak ada rules. Turunkan parameter Min Support/Confidence.")
    else:
        st.markdown('<p class="section-title">Cari Rekomendasi Produk</p>', unsafe_allow_html=True)

        # Get all antecedent products
        all_ant_products = sorted(set(
            item for antset in rules_df["antecedents"] for item in antset
        ))

        search_col, n_col = st.columns([3, 1])
        search_q = search_col.text_input("🔍 Ketik nama produk (sebagian kata sudah cukup)", placeholder="contoh: LUNCH BAG, ROSES, REGENCY...")
        top_rec = n_col.number_input("Jumlah rekomendasi", 3, 20, 5, step=1)

        if search_q:
            q_upper = search_q.upper().strip()
            matched_rules = rules_df[
                rules_df["antecedents"].apply(
                    lambda x: any(q_upper in item.upper() for item in x)
                )
            ].sort_values("lift", ascending=False).head(top_rec)

            if matched_rules.empty:
                st.warning(f"Tidak ditemukan rules untuk kata kunci **'{search_q}'**.")
                st.markdown("**Produk tersedia (top 20 sebagai antecedent):**")
                top_ants = rules_df["antecedents"].explode().value_counts().head(20)
                badges = "".join([f'<span class="badge">{p}</span>' for p in top_ants.index])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.markdown(f"**{len(matched_rules)} rekomendasi ditemukan untuk:** `{search_q.upper()}`")
                for i, (_, row) in enumerate(matched_rules.iterrows(), 1):
                    ant = ", ".join(list(row["antecedents"]))
                    con = ", ".join(list(row["consequents"]))
                    conf_pct = row["confidence"] * 100
                    bar_width = int(conf_pct)
                    st.markdown(f"""
                    <div class="rule-card">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <span style="color:#8B949E;font-size:0.75rem">#{i}</span>
                            <span style="color:#3FB950;font-family:'Space Mono',monospace;font-size:0.8rem">Lift {row['lift']:.1f}x</span>
                        </div>
                        <div style="margin:0.4rem 0">
                            <span class="ant">📦 {ant}</span>
                        </div>
                        <div style="color:#8B949E;font-size:0.8rem;margin:0.2rem 0">↓ sering dibeli bersama dengan</div>
                        <div style="margin:0.4rem 0">
                            <span class="con">✅ {con}</span>
                        </div>
                        <div style="margin-top:0.6rem">
                            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#8B949E;margin-bottom:0.25rem">
                                <span>Confidence</span><span>{conf_pct:.1f}%</span>
                            </div>
                            <div style="background:#21262D;border-radius:4px;height:6px">
                                <div style="background:linear-gradient(90deg,#58A6FF,#3FB950);width:{bar_width}%;height:6px;border-radius:4px"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("**Top Produk sebagai Antecedent:**")
            top_ants = rules_df["antecedents"].explode().value_counts().head(30)
            badges = "".join([f'<span class="badge">{p}</span>' for p in top_ants.index])
            st.markdown(badges, unsafe_allow_html=True)
            st.info("💡 Klik salah satu nama produk di atas dan ketikkan di kolom pencarian.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VISUALISASI
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if rules_df.empty:
        st.warning("⚠️ Tidak ada rules untuk divisualisasikan.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-title">Support vs Confidence (ukuran = Lift)</p>', unsafe_allow_html=True)
            fig_sc = px.scatter(
                rules_df.head(100),
                x="support", y="confidence",
                size="lift", color="lift",
                hover_data={"antecedents": rules_df.head(100)["antecedents"].apply(lambda x: ", ".join(list(x))),
                            "consequents": rules_df.head(100)["consequents"].apply(lambda x: ", ".join(list(x))),
                            "lift": ":.2f"},
                color_continuous_scale=["#1F6FEB", "#58A6FF", "#3FB950", "#F78166"],
                labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"}
            )
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=380, coloraxis_colorbar=dict(title="Lift", tickfont=dict(color="#8B949E")))
            st.plotly_chart(fig_sc, use_container_width=True)

        with col2:
            st.markdown('<p class="section-title">Distribusi Lift Score</p>', unsafe_allow_html=True)
            fig_lift = px.histogram(
                rules_df, x="lift", nbins=40,
                labels={"lift": "Lift Score", "count": "Jumlah Rules"},
                color_discrete_sequence=["#3FB950"]
            )
            fig_lift.update_layout(**PLOTLY_LAYOUT, height=380, bargap=0.05)
            st.plotly_chart(fig_lift, use_container_width=True)

        # Heatmap: top antecedents vs lift
        st.markdown('<p class="section-title">Top Rules — Antecedent vs Lift (Heatmap)</p>', unsafe_allow_html=True)
        top_rules_vis = rules_df.head(20).copy()
        top_rules_vis["ant_str"] = top_rules_vis["antecedents"].apply(lambda x: ", ".join(list(x))[:40])
        top_rules_vis["con_str"] = top_rules_vis["consequents"].apply(lambda x: ", ".join(list(x))[:40])
        top_rules_vis["label"] = top_rules_vis["ant_str"] + " → " + top_rules_vis["con_str"]

        fig_bar = px.bar(
            top_rules_vis,
            x="lift", y="label",
            orientation="h",
            color="confidence",
            color_continuous_scale=["#1F6FEB", "#58A6FF", "#3FB950"],
            labels={"lift": "Lift", "label": "", "confidence": "Confidence"},
            hover_data={"support": ":.4f", "confidence": ":.4f", "lift": ":.2f"}
        )
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=500, yaxis=dict(autorange="reversed"),
                              coloraxis_colorbar=dict(title="Conf", tickfont=dict(color="#8B949E")))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Frequent itemsets support distribution
        if not freq_items.empty:
            st.markdown('<p class="section-title">Top 20 Frequent Itemsets (2-item) berdasarkan Support</p>', unsafe_allow_html=True)
            pairs = freq_items[freq_items["length"] == 2].sort_values("support", ascending=False).head(20)
            pairs["label"] = pairs["itemsets"].apply(lambda x: " + ".join(list(x))[:55])
            fig_pairs = px.bar(
                pairs, x="support", y="label",
                orientation="h",
                color="support",
                color_continuous_scale=["#1F2933", "#D2A8FF"],
                labels={"support": "Support", "label": ""}
            )
            fig_pairs.update_layout(**PLOTLY_LAYOUT, height=480, yaxis=dict(autorange="reversed"),
                                    coloraxis_showscale=False)
            st.plotly_chart(fig_pairs, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Raw Data (cleaned)</p>', unsafe_allow_html=True)

    country_filter = st.selectbox("Filter Negara", ["All"] + sorted(df_clean["Country"].unique().tolist()), key="exp_country")
    df_view = df_clean if country_filter == "All" else df_clean[df_clean["Country"] == country_filter]

    col_r, col_p = st.columns(2)
    col_r.metric("Baris ditampilkan", f"{len(df_view):,}")
    col_p.metric("Produk unik", f"{df_view['Description'].nunique():,}")

    st.dataframe(
        df_view[["InvoiceNo", "Description", "Quantity", "UnitPrice", "CustomerID", "Country", "InvoiceDate"]]
        .head(500)
        .reset_index(drop=True),
        use_container_width=True,
        height=340
    )
    st.caption("Menampilkan maksimal 500 baris pertama.")

    if not rules_df.empty:
        st.markdown('<p class="section-title" style="margin-top:1.5rem">Semua Association Rules</p>', unsafe_allow_html=True)
        rules_export = rules_df.copy()
        rules_export["antecedents"] = rules_export["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_export["consequents"] = rules_export["consequents"].apply(lambda x: ", ".join(list(x)))
        rules_export = rules_export[["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]]
        rules_export = rules_export.round(4)
        st.dataframe(rules_export, use_container_width=True, height=340)

        csv = rules_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Rules sebagai CSV",
            data=csv,
            file_name="association_rules.csv",
            mime="text/csv"
        )
