# ─────────────────────────────────────────────────────────────────────────
#  💎 Diamond Dynamics — Price Prediction & Market Segmentation
# ─────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size:44px; font-weight:800; text-align:center;
        background:linear-gradient(135deg,#667eea,#764ba2);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        padding:10px 0 4px 0;
    }
    .sub-title { font-size:17px; color:#95a5a6; text-align:center; margin-bottom:10px; }
    .section-header {
        font-size:16px; font-weight:700; color:#2c3e50;
        padding:8px 14px; background:#f4f6f8;
        border-radius:8px; border-left:4px solid #667eea; margin-bottom:10px;
    }
    .inr-box {
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        border-radius:18px; padding:32px 20px; text-align:center;
        color:white; margin-top:15px; box-shadow:0 8px 28px rgba(102,126,234,0.4);
    }
    .inr-box .box-label { font-size:13px; font-weight:600; letter-spacing:2px; text-transform:uppercase; opacity:0.85; }
    .inr-box .inr-value { font-size:50px; font-weight:800; margin:10px 0 0 0; letter-spacing:-1px; }
    .usd-box {
        background:linear-gradient(135deg,#2c3e50 0%,#3498db 100%);
        border-radius:18px; padding:24px 20px; text-align:center;
        color:white; margin-top:12px; box-shadow:0 6px 20px rgba(52,152,219,0.35);
    }
    .usd-box .box-label { font-size:13px; font-weight:600; letter-spacing:2px; text-transform:uppercase; opacity:0.85; }
    .usd-box .usd-value { font-size:36px; font-weight:800; margin:8px 0 0 0; }
    .cluster-box {
        background:linear-gradient(135deg,#11998e 0%,#38ef7d 100%);
        border-radius:18px; padding:32px 20px; text-align:center;
        color:white; margin-top:15px; box-shadow:0 8px 28px rgba(17,153,142,0.4);
    }
    .cluster-box .box-label { font-size:13px; font-weight:600; letter-spacing:2px; text-transform:uppercase; opacity:0.85; }
    .cluster-box .cluster-num { font-size:18px; font-weight:600; opacity:0.9; margin:10px 0 4px 0; }
    .cluster-box .cluster-name { font-size:30px; font-weight:800; margin:0; }
    .error-box {
        background:linear-gradient(135deg,#e74c3c,#c0392b);
        border-radius:15px; padding:20px; text-align:center; color:white; font-size:15px; margin-top:15px;
    }
    .insight-card {
        background:white; border-radius:12px; padding:16px; text-align:center;
        box-shadow:0 2px 12px rgba(0,0,0,0.07); border-top:3px solid #667eea;
    }
    .insight-card .i-label { font-size:11px; color:#95a5a6; text-transform:uppercase; letter-spacing:1px; }
    .insight-card .i-value { font-size:20px; font-weight:700; color:#2c3e50; margin-top:5px; }
    .seg-desc-box {
        background:#f0faf8; border:1px solid #38ef7d;
        border-radius:12px; padding:14px 18px; margin-top:12px;
    }
    .seg-desc-box .sd-title { font-size:14px; font-weight:700; color:#11998e; margin-bottom:5px; }
    .seg-desc-box .sd-text { font-size:14px; color:#2c3e50; }
    .metric-card {
        background:white; border-radius:10px; padding:18px;
        box-shadow:0 2px 12px rgba(0,0,0,0.07); text-align:center;
    }
    .stButton>button {
        width:100%;
        background:linear-gradient(135deg,#667eea,#764ba2) !important;
        color:white !important; border:none !important;
        border-radius:12px !important; padding:13px !important;
        font-size:17px !important; font-weight:700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
BASE_PATH   = r"D:\AI&ML\Diamond Dynamics"
MODELS_PATH = os.path.join(BASE_PATH, 'models')
USD_TO_INR  = 83.5

# ── Load Models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    loaded, errors = {}, []
    files = {
        'regressor'        : 'random_forest.pkl',
        'scaler'           : 'scaler.pkl',
        'feature_list'     : 'feature_list.pkl',
        'kmeans'           : 'kmeans_model.pkl',
        'scaler_cluster'   : 'scaler_cluster.pkl',
        'cluster_names'    : 'cluster_names.pkl',
        'cluster_features' : 'cluster_features.pkl',
    }
    for key, fname in files.items():
        try:
            with open(os.path.join(MODELS_PATH, fname), 'rb') as f:
                loaded[key] = pickle.load(f)
        except FileNotFoundError:
            errors.append(f"Missing: {fname}")
        except Exception as e:
            errors.append(f"Error — {fname}: {e}")
    return loaded, errors

models, load_errors = load_all_models()

# ── Encoding Maps ─────────────────────────────────────────────────────────
CUT_ORDINAL     = {'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4}
COLOR_ORDINAL   = {'J':0,'I':1,'H':2,'G':3,'F':4,'E':5,'D':6}
CLARITY_ORDINAL = {'I1':0,'SI2':1,'SI1':2,'VS2':3,'VS1':4,'VVS2':5,'VVS1':6,'IF':7}

CLUSTER_DESC = {
    '💎 Premium Heavy Diamonds'      : 'Large, expensive, premium-grade stones. Top-tier investment pieces.',
    '🧡 Mid-Range Balanced Diamonds' : 'Balanced size and price. Popular for engagement rings and gifts.',
    '💚 Affordable Small Diamonds'   : 'Small, budget-friendly stones. Perfect for casual jewellery.',
    '👑 High Quality Value Diamonds' : 'Great cut quality at a reasonable price. Best value-for-money.',
    '🔶 Large Budget Diamonds'       : 'Large size but lower quality. Buyers prioritising size.',
    '✨ Petite Luxury Diamonds'      : 'Small but flawless. Perfect for minimalist luxury jewellery.',
    '🟡 Economy Grade Diamonds'      : 'Moderate size, lower quality. Ideal for decorative use.',
    '🏆 Ultra Luxury Diamonds'       : 'Exceptional size and quality. Rare collector-grade diamonds.',
}

# ── Feature Builder — 9 features only ────────────────────────────────────
def build_features(carat, cut, color, clarity, depth, table, x, y, z):
    # Step 3: log1p on skewed features (same as training)
    carat_t = np.log1p(carat)
    x_t     = np.log2(x)
    y_t     = np.log2(y)
    z_t     = np.log2(z)
    depth_t     = np.log1p(depth)
    table_t     = np.log1p(table)

    return pd.DataFrame([{
        'carat'   : carat_t,
        'cut'     : CUT_ORDINAL[cut],
        'color'   : COLOR_ORDINAL[color],
        'clarity' : CLARITY_ORDINAL[clarity],
        'depth'   : depth_t,
        'table'   : table_t,
        'x'       : x_t,
        'y'       : y_t,
        'z'       : z_t,
    }])

# ── Price Prediction ──────────────────────────────────────────────────────
# VERIFIED: stored = log1p(USD x 83.5) x 83.5
# Reverse:  USD = expm1(raw / 83.5) / 83.5
def predict_price(carat, cut, color, clarity, depth, table, x, y, z):
    try:
        df      = build_features(carat, cut, color, clarity, depth, table, x, y, z)
        cols    = models['feature_list']
        aligned = df.reindex(columns=cols, fill_value=0).fillna(0)
        scaled  = models['scaler'].transform(aligned)
        raw     = models['regressor'].predict(scaled)[0]

        usd = np.expm1(raw / USD_TO_INR) / USD_TO_INR
        inr = usd * USD_TO_INR

        if np.isinf(usd) or np.isnan(usd) or usd <= 0:
            return None, None, f"Invalid prediction (raw={raw:.3f})"
        return float(inr), float(usd), None
    except Exception as e:
        return None, None, str(e)

# ── Cluster Prediction ────────────────────────────────────────────────────
def predict_cluster(carat, cut, color, clarity, depth, table, x, y, z):
    try:
        df      = build_features(carat, cut, color, clarity, depth, table, x, y, z)
        cols    = models['cluster_features']
        aligned = df.reindex(columns=cols, fill_value=0).fillna(0)
        scaled  = models['scaler_cluster'].transform(aligned)
        cid     = int(models['kmeans'].predict(scaled)[0])
        name    = models['cluster_names'].get(cid, f"Cluster {cid}")
        return cid, name, None
    except Exception as e:
        return None, None, str(e)

# ── Input Form ────────────────────────────────────────────────────────────
def render_form(prefix):
    st.markdown('<p class="section-header">🔢 Numerical Attributes</p>', unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        carat = st.number_input("💎 Carat Weight", min_value=0.20, max_value=5.00,
                                 value=0.90, step=0.01, key=f"{prefix}_carat",
                                 help="Weight of diamond (0.2–5.0 carats)")
        depth = st.number_input("📐 Depth (%)", min_value=43.0, max_value=79.0,
                                 value=61.7, step=0.1, key=f"{prefix}_depth",
                                 help="Total depth % = z/mean(x,y)×100")
    with n2:
        table = st.number_input("📊 Table (%)", min_value=43.0, max_value=95.0,
                                 value=57.5, step=0.1, key=f"{prefix}_table",
                                 help="Width of top facet as % of diameter")
        x = st.number_input("📏 Length — x (mm)", min_value=0.10, max_value=11.0,
                              value=6.30, step=0.01, key=f"{prefix}_x")
    with n3:
        y = st.number_input("📏 Width — y (mm)", min_value=0.10, max_value=11.0,
                              value=6.30, step=0.01, key=f"{prefix}_y")
        z = st.number_input("📏 Depth — z (mm)", min_value=0.10, max_value=7.00,
                              value=3.90, step=0.01, key=f"{prefix}_z")
    st.markdown("")
    st.markdown('<p class="section-header">🔤 Quality Attributes (The 4Cs)</p>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        cut = st.selectbox("✂️ Cut Quality", ['Fair','Good','Very Good','Premium','Ideal'],
                            index=4, key=f"{prefix}_cut", help="Ideal=best | Fair=lowest")
    with q2:
        color = st.selectbox("🎨 Color Grade", ['D','E','F','G','H','I','J'],
                              index=3, key=f"{prefix}_color", help="D=colorless(best) | J=noticeable")
    with q3:
        clarity = st.selectbox("🔬 Clarity Grade",
                                ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'],
                                index=3, key=f"{prefix}_clarity", help="IF=Flawless(best) | I1=visible")
    return carat, depth, table, x, y, z, cut, color, clarity

# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">💎 Diamond Dynamics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Price Prediction & Market Segmentation</p>', unsafe_allow_html=True)
st.markdown("---")

if load_errors:
    for e in load_errors:
        st.error(e)
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Diamond Dynamics")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("", ["🏠  Home","💰  Price Prediction","🔍  Market Segmentation","📊  Combined Analysis"])
    st.markdown("---")
    st.markdown("### ✅ Model Status")
    for label, key in {
        'Regressor':'regressor','Reg Scaler':'scaler','Feature List':'feature_list',
        'KMeans':'kmeans','Cluster Scaler':'scaler_cluster',
        'Cluster Names':'cluster_names','Cluster Feats':'cluster_features',
    }.items():
        st.markdown(f"{'✅' if key in models else '❌'} {label}")
    st.markdown("---")
    st.info("📦 Dataset: 53,940 diamonds\n\n🤖 Model: Random Forest\n\n📊 R² Score: 0.9924")

# ═══════════════════════════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    c1,c2,c3 = st.columns(3)
    for col,num,lbl in zip([c1,c2,c3],["53,940","0.9924","4+"],
                            ["Diamonds Trained","R² Score","Market Segments"]):
        col.markdown(f"""<div class="metric-card">
            <h2 style="color:#667eea">{num}</h2>
            <p style="color:#7f8c8d;margin:0">{lbl}</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.success("**💰 Price Prediction**\n- Input 9 diamond attributes\n- Primary: **INR** | Secondary: **USD**\n- Random Forest (R²=0.9924)")
    with c2:
        st.info("**🔍 Market Segmentation**\n- Same 9-attribute input\n- Returns cluster number + name\n- K-Means Clustering")
    st.markdown("### 📊 The 4C's Reference")
    st.dataframe(pd.DataFrame({
        'Feature':['Carat','Cut','Color','Clarity'],
        'Best':['Higher','Ideal','D','IF'],
        'Worst':['Lower','Fair','J','I1'],
        'Price Impact':['⭐⭐⭐⭐⭐','⭐⭐⭐⭐','⭐⭐⭐','⭐⭐⭐'],
    }), use_container_width=True, hide_index=True)
    st.markdown("### 🏷️ Market Segments")
    st.dataframe(pd.DataFrame([{'Segment':k,'Description':v} for k,v in CLUSTER_DESC.items()]),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PRICE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💰  Price Prediction":
    st.markdown("## 💰 Diamond Price Prediction")
    st.markdown("Fill in all attributes and click **Predict Price**.")
    st.markdown("---")
    with st.form("price_form"):
        carat,depth,table,x,y,z,cut,color,clarity = render_form("price")
        st.markdown("")
        btn = st.form_submit_button("💰  Predict Price")
    if btn:
        with st.spinner("🔮 Predicting..."):
            inr,usd,err = predict_price(carat,cut,color,clarity,depth,table,x,y,z)
        if err:
            st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="inr-box">
                <div class="box-label">💎 Predicted Diamond Price</div>
                <div class="inr-value">₹ {inr:,.2f}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="usd-box">
                <div class="box-label">💵 Price in US Dollars</div>
                <div class="usd-value">$ {usd:,.2f} USD</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("### 💡 Price Breakdown")
            i1,i2,i3,i4 = st.columns(4)
            for col,lbl,val in zip([i1,i2,i3,i4],
                ["Price (INR)","Price (USD)","Price (EUR)","Per Carat (INR)"],
                [f"₹{inr:,.0f}",f"${usd:,.2f}",f"€{usd*0.92:,.2f}",f"₹{inr/carat:,.0f}"]):
                col.markdown(f"""<div class="insight-card">
                    <div class="i-label">{lbl}</div>
                    <div class="i-value">{val}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("### 📋 Input Summary")
            cols = st.columns(9)
            for c,l,v in zip(cols,["Carat","Cut","Color","Clarity","Depth%","Table%","x(mm)","y(mm)","z(mm)"],
                             [carat,cut,color,clarity,depth,table,x,y,z]):
                c.metric(l, f"{v}")

# ═══════════════════════════════════════════════════════════════════════════
#  MARKET SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍  Market Segmentation":
    st.markdown("## 🔍 Diamond Market Segmentation")
    st.markdown("Fill in all attributes and click **Predict Cluster**.")
    st.markdown("---")
    with st.form("cluster_form"):
        carat,depth,table,x,y,z,cut,color,clarity = render_form("cluster")
        st.markdown("")
        btn = st.form_submit_button("🔍  Predict Cluster")
    if btn:
        with st.spinner("🔮 Identifying segment..."):
            cid,cname,err = predict_cluster(carat,cut,color,clarity,depth,table,x,y,z)
        if err:
            st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="cluster-box">
                <div class="box-label">🔍 Market Segment Identified</div>
                <div class="cluster-num">Cluster Number : {cid}</div>
                <div class="cluster-name">{cname}</div>
            </div>""", unsafe_allow_html=True)
            desc = CLUSTER_DESC.get(cname,"A distinct diamond market segment.")
            st.markdown(f"""<div class="seg-desc-box">
                <div class="sd-title">📌 About this Segment</div>
                <div class="sd-text">{desc}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("### 📊 All Market Segments")
            st.dataframe(pd.DataFrame([{'Segment':k,'Description':v,
                'Your Diamond':'✅ Match!' if k==cname else ''} for k,v in CLUSTER_DESC.items()]),
                use_container_width=True, hide_index=True)
            st.markdown("")
            st.markdown("### 📋 Input Summary")
            cols = st.columns(9)
            for c,l,v in zip(cols,["Carat","Cut","Color","Clarity","Depth%","Table%","x(mm)","y(mm)","z(mm)"],
                             [carat,cut,color,clarity,depth,table,x,y,z]):
                c.metric(l, f"{v}")

# ═══════════════════════════════════════════════════════════════════════════
#  COMBINED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊  Combined Analysis":
    st.markdown("## 📊 Combined Diamond Analysis")
    st.markdown("Get **Price** AND **Market Segment** together!")
    st.markdown("---")
    with st.form("combined_form"):
        carat,depth,table,x,y,z,cut,color,clarity = render_form("combined")
        st.markdown("")
        btn = st.form_submit_button("🚀  Analyse Diamond")
    if btn:
        with st.spinner("🔮 Running full analysis..."):
            inr,usd,perr   = predict_price(carat,cut,color,clarity,depth,table,x,y,z)
            cid,cname,cerr = predict_cluster(carat,cut,color,clarity,depth,table,x,y,z)
        r1,r2 = st.columns(2)
        with r1:
            if perr:
                st.markdown(f'<div class="error-box">❌ {perr}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="inr-box">
                    <div class="box-label">💎 Predicted Price</div>
                    <div class="inr-value">₹ {inr:,.2f}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="usd-box">
                    <div class="box-label">💵 Price in USD</div>
                    <div class="usd-value">$ {usd:,.2f} USD</div>
                </div>""", unsafe_allow_html=True)
        with r2:
            if cerr:
                st.markdown(f'<div class="error-box">❌ {cerr}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="cluster-box">
                    <div class="box-label">🔍 Market Segment</div>
                    <div class="cluster-num">Cluster № {cid}</div>
                    <div class="cluster-name">{cname}</div>
                </div>""", unsafe_allow_html=True)
                desc = CLUSTER_DESC.get(cname,"")
                st.markdown(f"""<div class="seg-desc-box">
                    <div class="sd-title">📌 Segment Info</div>
                    <div class="sd-text">{desc}</div>
                </div>""", unsafe_allow_html=True)
        if inr:
            st.markdown("")
            st.markdown("### 💡 Price Breakdown")
            i1,i2,i3,i4 = st.columns(4)
            for col,lbl,val in zip([i1,i2,i3,i4],["INR","USD","EUR","Per Carat"],
                [f"₹{inr:,.0f}",f"${usd:,.2f}",f"€{usd*0.92:,.2f}",f"₹{inr/carat:,.0f}"]):
                col.markdown(f"""<div class="insight-card">
                    <div class="i-label">{lbl}</div>
                    <div class="i-value">{val}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("")
        st.markdown("### 📋 Diamond Report Card")
        cols = st.columns(9)
        for c,l,v in zip(cols,["Carat","Cut","Color","Clarity","Depth%","Table%","x(mm)","y(mm)","z(mm)"],
                         [carat,cut,color,clarity,depth,table,x,y,z]):
            c.metric(l, f"{v}")

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#bdc3c7;font-size:13px;'>💎 Diamond Dynamics &nbsp;|&nbsp; Price Prediction & Market Segmentation &nbsp;|&nbsp; Built with Streamlit · Scikit-learn · K-Means</p>", unsafe_allow_html=True)