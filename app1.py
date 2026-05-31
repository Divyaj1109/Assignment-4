# ─────────────────────────────────────────────────────────────────────────
#  💎 Diamond Dynamics — Price Prediction & Market Segmentation App
# ─────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title = "💎 Diamond Dynamics",
    page_icon  = "💎",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 42px; font-weight: bold;
            color: #2c3e50; text-align: center; padding: 10px 0;
        }
        .sub-title {
            font-size: 18px; color: #7f8c8d;
            text-align: center; margin-bottom: 30px;
        }
        .result-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 15px; padding: 30px;
            text-align: center; color: white;
            font-size: 28px; font-weight: bold; margin-top: 20px;
        }
        .cluster-box {
            background: linear-gradient(135deg, #11998e, #38ef7d);
            border-radius: 15px; padding: 30px;
            text-align: center; color: white;
            font-size: 24px; font-weight: bold; margin-top: 20px;
        }
        .error-box {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            border-radius: 15px; padding: 20px;
            text-align: center; color: white;
            font-size: 18px; margin-top: 20px;
        }
        .info-box {
            background-color: #f0f3f4;
            border-left: 5px solid #2980b9;
            border-radius: 5px; padding: 15px; margin: 10px 0;
        }
        .metric-card {
            background: black; border-radius: 10px; padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center;
        }
        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; border: none; border-radius: 10px;
            padding: 12px; font-size: 18px; font-weight: bold;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #764ba2, #667eea);
        }
    </style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────
BASE_PATH   = r"D:\AI&ML\Diamond Dynamics"
MODELS_PATH = os.path.join(BASE_PATH, 'models')

# ✅ Set True if you applied np.log1p(price) during training
# ✅ Set False if price was only converted USD → INR (no log)
LOG_TRANSFORM_APPLIED = False


# ── Load Models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    loaded = {}
    errors = []

    files = {
        'regressor'       : 'random_forest.pkl',
        'scaler'          : 'scaler.pkl',
        'kmeans'          : 'kmeans_model.pkl',
        'scaler_cluster'  : 'scaler_cluster.pkl',
        'cluster_names'   : 'cluster_names.pkl',
        'cluster_features': 'cluster_features.pkl',
    }

    for key, filename in files.items():
        path = os.path.join(MODELS_PATH, filename)
        try:
            with open(path, 'rb') as f:
                loaded[key] = pickle.load(f)
        except FileNotFoundError:
            errors.append(f"❌ Missing: {filename}")
        except Exception as e:
            errors.append(f"❌ Error loading {filename}: {e}")

    return loaded, errors

models, load_errors = load_models()


# ── Encoding Maps ─────────────────────────────────────────────────────────
cut_map = {
    'Fair': 0, 'Good': 1, 'Very Good': 2,
    'Premium': 3, 'Ideal': 4
}
color_map = {
    'J': 0, 'I': 1, 'H': 2, 'G': 3,
    'F': 4, 'E': 5, 'D': 6
}
clarity_map = {
    'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3,
    'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7
}

cluster_descriptions = {
    '💎 Premium Heavy Diamonds'      : 'Large, expensive, premium-grade stones. Top-tier investment pieces for luxury buyers.',
    '🧡 Mid-Range Balanced Diamonds' : 'Balanced size and price. Popular choice for engagement rings and gifts.',
    '💚 Affordable Small Diamonds'   : 'Small, budget-friendly stones. Perfect for casual jewellery and first-time buyers.',
    '👑 High Quality Value Diamonds' : 'Great cut quality at a reasonable price. Best value-for-money segment.',
    '🔶 Large Budget Diamonds'       : 'Large size but lower quality grade. Buyers prioritising size over cut/clarity.',
    '✨ Petite Luxury Diamonds'      : 'Small but flawless stones. Perfect for minimalist luxury jewellery.',
    '🟡 Economy Grade Diamonds'      : 'Moderate size with lower quality grades. Ideal for decorative use.',
    '🏆 Ultra Luxury Diamonds'       : 'Exceptional size and quality. Rare collector-grade investment diamonds.',
}


# ── Feature Builder ───────────────────────────────────────────────────────
def build_feature_vector(carat, cut, color, clarity,
                          depth, table, x, y, z):
    cut_enc     = cut_map[cut]
    color_enc   = color_map[color]
    clarity_enc = clarity_map[clarity]

    volume          = x * y * z
    dimension_ratio = (x + y) / (2 * z + 1e-9)
    surface_area    = 2 * (x*y + y*z + x*z)
    carat_per_vol   = carat / (volume + 1e-9)
    xy_symmetry     = min(x, y) / (max(x, y) + 1e-9)
    quality_score   = cut_enc + color_enc + clarity_enc

    features = {
        'carat'           : carat,
        'cut'             : cut_enc,
        'color'           : color_enc,
        'clarity'         : clarity_enc,
        'depth'           : depth,
        'table'           : table,
        'x'               : x,
        'y'               : y,
        'z'               : z,
        'volume'          : volume,
        'price_per_carat' : 0,
        'dimension_ratio' : dimension_ratio,
        'surface_area'    : surface_area,
        'carat_per_volume': carat_per_vol,
        'xy_symmetry'     : xy_symmetry,
        'cut_score'       : cut_enc,
        'color_score'     : color_enc,
        'clarity_score'   : clarity_enc,
        'quality_score'   : quality_score,
    }
    return pd.DataFrame([features])


# ── Safe Price Prediction ─────────────────────────────────────────────────
def predict_price(input_df):
    try:
        # Align to model's expected features
        if hasattr(models['regressor'], 'feature_names_in_'):
            reg_cols = list(models['regressor'].feature_names_in_)
        else:
            reg_cols = list(input_df.columns)

        input_aligned = input_df.reindex(columns=reg_cols, fill_value=0)
        input_aligned = input_aligned.fillna(0)

        # Scale
        input_scaled = models['scaler'].transform(input_aligned)

        # Raw prediction
        raw = models['regressor'].predict(input_scaled)[0]

        # Auto-detect log transform
        # If raw value is small (< 20), it's likely log-transformed
        if LOG_TRANSFORM_APPLIED or raw < 20:
            price = np.expm1(raw)
        else:
            price = raw

        # Validate result
        if np.isinf(price) or np.isnan(price) or price <= 0:
            return None, f"Invalid prediction value: {raw}. " \
                          f"Check LOG_TRANSFORM_APPLIED flag in app.py"

        return float(price), None

    except Exception as e:
        return None, str(e)


# ── Safe Cluster Prediction ───────────────────────────────────────────────
def predict_cluster(input_df):
    try:
        cluster_cols  = models['cluster_features']
        input_cluster = input_df.reindex(columns=cluster_cols, fill_value=0)
        input_cluster = input_cluster.fillna(0)

        input_scaled = models['scaler_cluster'].transform(input_cluster)
        cluster_id   = int(models['kmeans'].predict(input_scaled)[0])
        cluster_name = models['cluster_names'].get(
            cluster_id, f"Cluster {cluster_id}"
        )
        return cluster_id, cluster_name, None

    except Exception as e:
        return None, None, str(e)


# ── Input Form ────────────────────────────────────────────────────────────
def render_input_form(key_prefix=""):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔢 Numerical Features**")
        carat = st.number_input("Carat Weight",
                                 min_value=0.20, max_value=5.00,
                                 value=0.90, step=0.01,
                                 help="Weight of the diamond (0.2–5.0)",
                                 key=f"{key_prefix}_carat")
        depth = st.number_input("Depth (%)",
                                 min_value=43.0, max_value=79.0,
                                 value=61.5, step=0.1,
                                 help="Total depth percentage",
                                 key=f"{key_prefix}_depth")
        table = st.number_input("Table (%)",
                                 min_value=43.0, max_value=95.0,
                                 value=57.0, step=0.1,
                                 help="Width of top facet %",
                                 key=f"{key_prefix}_table")
    with col2:
        st.markdown("**📐 Dimensions (mm)**")
        x = st.number_input("Length (x mm)",
                              min_value=0.10, max_value=11.0,
                              value=4.50, step=0.01,
                              key=f"{key_prefix}_x")
        y = st.number_input("Width  (y mm)",
                              min_value=0.10, max_value=11.0,
                              value=4.50, step=0.01,
                              key=f"{key_prefix}_y")
        z = st.number_input("Depth  (z mm)",
                              min_value=0.10, max_value=7.00,
                              value=2.80, step=0.01,
                              key=f"{key_prefix}_z")
    with col3:
        st.markdown("**🔤 Quality Features**")
        cut = st.selectbox("Cut Quality",
                            ['Fair','Good','Very Good','Premium','Ideal'],
                            index=4,
                            help="Ideal is the best cut quality",
                            key=f"{key_prefix}_cut")
        color = st.selectbox("Color Grade",
                              ['D','E','F','G','H','I','J'],
                              index=3,
                              help="D is the best color grade",
                              key=f"{key_prefix}_color")
        clarity = st.selectbox("Clarity Grade",
                                ['IF','VVS1','VVS2','VS1',
                                 'VS2','SI1','SI2','I1'],
                                index=3,
                                help="IF = Internally Flawless (best)",
                                key=f"{key_prefix}_clarity")

    return carat, depth, table, x, y, z, cut, color, clarity


# ═══════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">💎 Diamond Dynamics</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Price Prediction & Market Segmentation</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Model Load Errors ─────────────────────────────────────────────────────
if load_errors:
    for err in load_errors:
        st.error(err)
    st.warning("⚠️ Some models failed to load. "
               "Check your models/ folder and file names.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Diamond Dynamics")
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    page = st.radio("Select Module:", [
        "🏠 Home",
        "💰 Price Prediction",
        "🔍 Market Segmentation",
        "📊 Combined Analysis"
    ])
    st.markdown("---")

    # Model status
    st.markdown("### ✅ Model Status")
    model_status = {
        'Regressor'      : 'regressor',
        'Reg Scaler'     : 'scaler',
        'KMeans'         : 'kmeans',
        'Cluster Scaler' : 'scaler_cluster',
        'Cluster Names'  : 'cluster_names',
        'Cluster Feats'  : 'cluster_features',
    }
    for label, key in model_status.items():
        icon = "✅" if key in models else "❌"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.info("Built with Streamlit & Scikit-learn\n\n"
            "Dataset: 53,940 diamonds")


# ── HOME PAGE ─────────────────────────────────────────────────────────────
if page == "🏠 Home":

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='metric-card'>
            <h2>53,940</h2><p>Diamonds Trained On</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <h2>6 Models</h2><p>ML + ANN Regression</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
            <h2>4 Segments</h2><p>Market Clusters</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.success("""
        **💰 Price Prediction**
        - Input diamond attributes
        - Get instant price in INR
        - Powered by Random Forest
        """)
    with c2:
        st.info("""
        **🔍 Market Segmentation**
        - Identify diamond market cluster
        - Named segments with descriptions
        - Powered by K-Means Clustering
        """)

    st.markdown("---")
    st.markdown("### 📊 The 4C's — Diamond Quality Guide")
    guide = pd.DataFrame({
        'Feature'      : ['Carat',     'Cut',    'Color', 'Clarity'],
        'Best Value'   : ['Higher',    'Ideal',  'D',     'IF'],
        'Worst Value'  : ['Lower',     'Fair',   'J',     'I1'],
        'Price Impact' : ['Very High', 'High',   'Medium','Medium']
    })
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown("### 🏷️ Market Segments")
    seg_df = pd.DataFrame([
        {'Segment': k, 'Description': v}
        for k, v in cluster_descriptions.items()
    ])
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ── PRICE PREDICTION PAGE ─────────────────────────────────────────────────
elif page == "💰 Price Prediction":
    st.markdown("## 💰 Diamond Price Prediction")
    st.markdown("Enter the diamond attributes below to predict its price in INR.")
    st.markdown("---")

    with st.form("price_form"):
        st.markdown("### 📝 Diamond Attributes")
        carat, depth, table, x, y, z, cut, color, clarity = \
            render_input_form("price")
        submitted = st.form_submit_button("💰 Predict Price")

    if submitted:
        with st.spinner("🔮 Predicting diamond price..."):

            input_df = build_feature_vector(
                carat, cut, color, clarity,
                depth, table, x, y, z
            )
            price, err = predict_price(input_df)

            if err:
                st.markdown(f"""<div class='error-box'>
                    ❌ Prediction Error<br>{err}
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class='result-box'>
                    💎 Predicted Diamond Price<br>
                    ₹ {price:,.2f} INR
                </div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("### 📋 Input Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Carat",   f"{carat}")
                m2.metric("Cut",      cut)
                m3.metric("Color",    color)
                m4.metric("Clarity",  clarity)

                m5, m6, m7 = st.columns(3)
                m5.metric("Volume (mm³)", f"{x*y*z:.2f}")
                m6.metric("Depth %",      f"{depth}")
                m7.metric("Table %",      f"{table}")

                # Price breakdown
                st.markdown("---")
                st.markdown("### 💡 Price Insights")
                i1, i2, i3 = st.columns(3)
                i1.info(f"**Price per Carat**\n\n₹ {price/carat:,.0f}")
                i2.info(f"**Price in USD**\n\n$ {price/83.5:,.2f}")
                i3.info(f"**Price in EUR**\n\n€ {price/90.5:,.2f}")


# ── MARKET SEGMENTATION PAGE ──────────────────────────────────────────────
elif page == "🔍 Market Segmentation":
    st.markdown("## 🔍 Diamond Market Segmentation")
    st.markdown("Identify which market segment your diamond belongs to.")
    st.markdown("---")

    with st.form("cluster_form"):
        st.markdown("### 📝 Diamond Attributes")
        carat, depth, table, x, y, z, cut, color, clarity = \
            render_input_form("cluster")
        submitted = st.form_submit_button("🔍 Predict Market Segment")

    if submitted:
        with st.spinner("🔮 Identifying market segment..."):

            input_df = build_feature_vector(
                carat, cut, color, clarity,
                depth, table, x, y, z
            )
            cluster_id, cluster_name, err = predict_cluster(input_df)

            if err:
                st.markdown(f"""<div class='error-box'>
                    ❌ Cluster Prediction Error<br>{err}
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class='cluster-box'>
                    🔍 Market Segment Identified!<br>
                    Cluster {cluster_id} — {cluster_name}
                </div>""", unsafe_allow_html=True)

                st.markdown("---")
                desc = cluster_descriptions.get(
                    cluster_name,
                    "A distinct diamond market segment."
                )
                st.markdown(f"""
                <div class='info-box'>
                    <b>📌 About this Segment:</b><br>{desc}
                </div>""", unsafe_allow_html=True)

                # All segments table
                st.markdown("### 📊 All Market Segments")
                all_segs = pd.DataFrame([
                    {
                        'Segment'     : k,
                        'Description' : v,
                        'Your Diamond': '✅ You are here!'
                                        if k == cluster_name else ''
                    }
                    for k, v in cluster_descriptions.items()
                ])
                st.dataframe(all_segs, use_container_width=True,
                             hide_index=True)


# ── COMBINED ANALYSIS PAGE ────────────────────────────────────────────────
elif page == "📊 Combined Analysis":
    st.markdown("## 📊 Combined Diamond Analysis")
    st.markdown("Get **price prediction** AND **market segment** together!")
    st.markdown("---")

    with st.form("combined_form"):
        st.markdown("### 📝 Diamond Attributes")
        carat, depth, table, x, y, z, cut, color, clarity = \
            render_input_form("combined")
        submitted = st.form_submit_button("🚀 Analyse Diamond")

    if submitted:
        with st.spinner("🔮 Running full diamond analysis..."):

            input_df = build_feature_vector(
                carat, cut, color, clarity,
                depth, table, x, y, z
            )

            price,      price_err   = predict_price(input_df)
            cluster_id, cluster_name, cluster_err = predict_cluster(input_df)

            # ── Results ───────────────────────────────────────────────
            r1, r2 = st.columns(2)

            with r1:
                if price_err:
                    st.markdown(f"""<div class='error-box'>
                        ❌ Price Error<br>{price_err}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-box'>
                        💰 Predicted Price<br>
                        ₹ {price:,.2f} INR
                    </div>""", unsafe_allow_html=True)

            with r2:
                if cluster_err:
                    st.markdown(f"""<div class='error-box'>
                        ❌ Cluster Error<br>{cluster_err}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='cluster-box'>
                        🔍 Market Segment<br>
                        {cluster_name}
                    </div>""", unsafe_allow_html=True)

            # ── Diamond Report Card ───────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Diamond Report Card")

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Carat",    f"{carat}")
            m2.metric("Cut",       cut)
            m3.metric("Color",     color)
            m4.metric("Clarity",   clarity)
            m5.metric("Volume",   f"{x*y*z:.2f}")
            m6.metric("Cluster",  f"#{cluster_id}"
                                   if cluster_id is not None else "—")

            # ── Segment Description ───────────────────────────────────
            if cluster_name:
                desc = cluster_descriptions.get(cluster_name, "")
                if desc:
                    st.markdown(f"""
                    <div class='info-box'>
                        <b>📌 Segment Info:</b> {desc}
                    </div>""", unsafe_allow_html=True)

            # ── Price Insights ────────────────────────────────────────
            if price:
                st.markdown("### 💡 Price Insights")
                i1, i2, i3 = st.columns(3)
                i1.info(f"**Price per Carat**\n\n₹ {price/carat:,.0f}")
                i2.info(f"**Price in USD**\n\n$ {price/83.5:,.2f}")
                i3.info(f"**Price in EUR**\n\n€ {price/90.5:,.2f}")


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#95a5a6; font-size:13px;'>
    💎 Diamond Dynamics | Price Prediction & Market Segmentation<br>
    Built with Streamlit · Scikit-learn · XGBoost · K-Means
</div>""", unsafe_allow_html=True)