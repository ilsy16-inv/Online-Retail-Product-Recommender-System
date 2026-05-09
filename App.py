import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# Mengabaikan peringatan agar tampilan UI tetap bersih
warnings.filterwarnings('ignore')

# ─── KONFIGURASI HALAMAN ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Basket Analysis — Online Retail",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS (THEME GITHUB DARK) ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0D1117;
    --surface: #161B22;
    --surface2: #21262D;
    --border: #30363D;
    --accent: #58A6FF;
    --accent2: #F78166;
    --accent3: #3FB950;
    --text: #E6EDF3;
    --muted: #8B949E;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Styling Card Metrik */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}

/* Styling Rule Card */
.rule-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─── FUNGSI PEMPROSESAN DATA ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_clean_data(file):
    """Membaca file Excel dan membersihkan data dasar."""
    try:
        df = pd.read_excel(file)
        # Pastikan kolom yang dibutuhkan ada
        required_columns = ["InvoiceNo", "Description", "Quantity", "Country"]
        if not all(col in df.columns for col in required_columns):
            return None, "Kolom tidak lengkap. Pastikan file memiliki kolom: InvoiceNo, Description, Quantity, Country."
        
        # Pembersihan
        df_clean = df.dropna(subset=["Description"])
        df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")] # Hapus Retur
        df_clean = df_clean[df_clean["Quantity"] > 0]
        df_clean["Description"] = df_clean["Description"].str.strip()
        
        return df_clean, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def generate_rules(df, country, min_sup, min_conf, max_len):
    """Menjalankan algoritma Apriori untuk mendapatkan association rules."""
    # Filter berdasarkan negara
    df_filtered = df[df["Country"] == country] if country != "All" else df
    
    # Transformasi data ke format list transaksi
    transactions = (
        df_filtered.groupby("InvoiceNo")["Description"]
        .apply(lambda x: list(set(x)))
        .tolist()
    )
    
    # Encoding transaksi
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Apriori
    freq_items = apriori(df_encoded, min_support=min_sup, use_colnames=True, max_len=max_len)
    
    if freq_items.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Association Rules
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    return freq_items, rules

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">📁 Unggah Data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih file Online_Retail.xlsx", type=["xlsx"])

    st.markdown('<p class="section-title">⚙️ Parameter Apriori</p>', unsafe_allow_html=True)
    min_support = st.slider("Min Support", 0.001, 0.10, 0.01, format="%.3f")
    min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3)
    max_items = st.slider("Maks Item per Set", 2, 5, 3)

    country_option = "All"
    if uploaded_file:
        df_clean, error = load_and_clean_data(uploaded_file)
        if not error:
            countries = ["All"] + sorted(df_clean["Country"].unique().tolist())
            country_option = st.selectbox("Pilih Negara", countries)

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.title("🛒 Market Basket Analysis")
st.markdown("Analisis keterkaitan produk menggunakan **Algoritma Apriori**.")

if not uploaded_file:
    st.info("Silakan unggah file Excel di sidebar untuk memulai.")
    st.stop()

if error:
    st.error(f"Error: {error}")
    st.stop()

# Eksekusi Algoritma
freq_items, rules = generate_rules(df_clean, country_option, min_support, min_confidence, max_items)

# Tampilan Tabs
tab1, tab2, tab3 = st.tabs(["📊 Statistik", "🔗 Association Rules", "📋 Data Raw"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transaksi", len(df_clean["InvoiceNo"].unique()))
    col2.metric("Produk Unik", len(df_clean["Description"].unique()))
    col3.metric("Rules Ditemukan", len(rules))

    st.markdown("### Top 10 Produk Terlaris")
    top_products = df_clean["Description"].value_counts().head(10)
    fig = px.bar(top_products, x=top_products.values, y=top_products.index, orientation='h',
                 labels={'x':'Frekuensi', 'index':''}, color_discrete_sequence=['#58A6FF'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#E6EDF3")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if rules.empty:
        st.warning("Tidak ada rules yang terbentuk. Coba turunkan nilai Support atau Confidence.")
    else:
        st.markdown(f"### Ditemukan {len(rules)} Association Rules")
        
        # Sortir rules berdasarkan Lift tertinggi
        rules_sorted = rules.sort_values("lift", ascending=False).head(20)
        
        for _, row in rules_sorted.iterrows():
            ant = ", ".join(list(row['antecedents']))
            con = ", ".join(list(row['consequents']))
            
            st.markdown(f"""
            <div class="rule-card">
                <p style="margin:0; font-size:0.9rem; color:#8B949E;">Jika membeli:</p>
                <p style="margin:0; font-weight:bold; color:#F78166;">{ant}</p>
                <p style="margin:5px 0; font-size:0.9rem; color:#8B949E;">Maka kemungkinan besar membeli:</p>
                <p style="margin:0; font-weight:bold; color:#3FB950;">{con}</p>
                <hr style="border-color:#30363D; margin:10px 0;">
                <div style="display:flex; gap:20px; font-family:'Space Mono', monospace; font-size:0.8rem;">
                    <span>Support: {row['support']:.3f}</span>
                    <span>Confidence: {row['confidence']:.3f}</span>
                    <span>Lift: {row['lift']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.dataframe(df_clean.head(100), use_container_width=True)
