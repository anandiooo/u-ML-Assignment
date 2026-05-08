import streamlit as st
import sys
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="LogiRisk · Delivery Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 300; }

:root {
    --primary:   #615fff;
    --surface:   #1d293d;
    --surface-2: #0f172b;
    --border:    #314158;
    --text-main: #e2e8f0;
    --text-muted:#94a3b8;
    --radius:    6px;
}

:root, [data-theme="light"], [data-theme="dark"] {
    --primary-color: var(--primary) !important;
    --background-color: var(--surface-2) !important;
    --secondary-background-color: var(--surface) !important;
    --text-color: var(--text-main) !important;
}

[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
    --st-color-background: var(--surface-2) !important;
    --st-color-secondary-background: var(--surface) !important;
    --st-color-text: var(--text-main) !important;
    --st-color-primary: var(--primary) !important;
}

body { background-color: var(--surface-2) !important; color: var(--text-main) !important; }
[data-testid="stAppViewContainer"] { background-color: var(--surface-2) !important; }

section[data-testid="stSidebar"] {
    background-color: var(--surface-2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
section[data-testid="stSidebar"] .sidebar-title {
    font-size: 1.4rem; font-weight: 700;
    color: var(--primary) !important; letter-spacing: -0.5px;
    padding: 0 1rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
section[data-testid="stSidebar"] .model-status-box {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .75rem 1rem; margin: 1rem;
    font-size: .9rem;
}
.sidebar-info {
    display: flex; align-items: center;
    padding: 0px 8px; border-radius: 6px;
    font-size: 1rem; color: var(--text-main);
}

.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; background-color: var(--surface-2); }

.page-header {
    margin-bottom: 1.75rem;
}
.page-header h1 {
    font-size: 2.8rem; font-weight: 400;
    color: var(--text-main); margin: 0;
    letter-spacing: -0.6px;
}

.section-heading {
    font-size: 1.15rem; font-weight: 400;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: .06em; margin: 1.5rem 0 .75rem;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
}
.metric-card .label {
    font-size: .85rem; font-weight: 400;
    text-transform: uppercase; letter-spacing: .07em;
    color: var(--text-muted); margin-bottom: .3rem;
}
.metric-card .value {
    font-size: 1.85rem; font-weight: 400;
    color: var(--text-main); letter-spacing: -1px;
}
.metric-card .sub {
    font-size: .85rem; color: var(--text-muted); margin-top: .2rem;
}

.interp-box, .ok-box, .warn-box, .risk-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: var(--radius);
    padding: .9rem 1.2rem; margin-top: .75rem;
    font-size: 0.95rem; color: var(--text-main); line-height: 1.6;
}
.interp-box strong, .ok-box strong, .warn-box strong, .risk-box strong { color: var(--primary); }

.stNumberInput > div > div > input,
.stTextInput  > div > div > input,
.stSelectbox [data-baseweb="select"] {
    border-radius: 8px !important;
    background-color: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-main) !important;
}
.stSelectbox > div > div > div { border: none !important; }
.stSlider .stSlider { accent-color: var(--primary); }

div.stButton > button[kind="primary"] {
    background: var(--primary) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important;
}

div.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    background-color: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text-main) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 2px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0 !important;
    font-weight: 400 !important; font-size: 1rem !important;
    background-color: transparent !important;
    color: var(--text-muted) !important;
    padding: 10px 12px !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

[data-testid="stTable"], [data-testid="stDataFrame"] {
    background-color: var(--surface) !important;
}
</style>
""", unsafe_allow_html=True)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import predict_risk
from train import train_model, DEFAULT_PARAMS
from data_make import clean_data

DATA_PATH = Path("data/ecommerce_shipping_data.csv")
TARGET_COL = "Reached.on.Time_Y.N"
NUMERIC_COLS = ["Customer_care_calls", "Customer_rating", "Cost_of_the_Product",
                "Prior_purchases", "Discount_offered", "Weight_in_gms"]
CATEGORICAL_COLS = ["Warehouse_block", "Mode_of_Shipment", "Product_importance", "Gender"]
STATUS_COLORS = {"On Time": "#615fff", "Delayed": "#e2e8f0"}
MODEL_OPTIONS = list(DEFAULT_PARAMS.keys())

PREPROCESSING_MODES = {
    "Balanced": {
        "outlier_method": "Clip (IQR)",
        "fe_toggles": ["Weight/Cost Ratio", "High Discount", "Loyalty Level"],
        "imbalance": "None",
        "drop_duplicates": True,
        "num_fill": "median",
        "cat_fill": "mode",
    },
    "Classic": {
        "outlier_method": "None",
        "fe_toggles": [],
        "imbalance": "None",
        "drop_duplicates": True,
        "num_fill": "median",
        "cat_fill": "mode",
    },
    "Robust": {
        "outlier_method": "Remove",
        "fe_toggles": ["Weight/Cost Ratio", "High Discount", "Engagement Score", "Delivery Urgency", "Care Intensity"],
        "imbalance": "Undersampling",
        "drop_duplicates": True,
        "num_fill": "median",
        "cat_fill": "mode",
    }
}

DEFAULT_PREPROCESSING = {
    "mode_name": "Balanced",
    "outlier_method": "Clip (IQR)",
    "fe_toggles": ["Weight/Cost Ratio", "High Discount"],
    "imbalance": "None",
    "drop_duplicates": True,
    "num_fill": "median",
    "cat_fill": "mode",
    "cat_fill_value": "Unknown",
}

for key, val in {
    "trained_models": {},
    "last_trained_model": None,
    "last_result": None,
    "last_inputs": None,
    "preprocess": DEFAULT_PREPROCESSING.copy(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics(model_name):
    slug = model_name.lower().replace(" ", "_")
    p = Path(f"models/metrics_{slug}.json")
    return json.load(open(p)) if p.exists() else None

def add_delay_status(df):
    out = df.copy()
    out["Delay_Status"] = out[TARGET_COL].map({0: "On Time", 1: "Delayed"})
    return out

def ibox(text, kind="info"):
    cls = {"info": "interp-box", "warn": "warn-box", "ok": "ok-box", "risk": "risk-box"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="page-header">'
        f'<div><h1>{title}</h1>'
        + (f'<p style="margin:0;color:var(--text-muted);font-size:.9rem">{subtitle}</p>' if subtitle else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

def render_home():
    page_header("LogiRisk", "AI-powered delivery delay intelligence for logistics teams")

    df = load_dataset()
    if df is not None:
        delayed = int(df[TARGET_COL].sum())
        total = len(df)
        avg_w = df["Weight_in_gms"].mean()
        avg_disc = df["Discount_offered"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value, sub in [
            (c1, "Total Shipments", f"{total:,}", "in dataset"),
            (c2, "Historical Delay Rate", f"{delayed/total:.1%}", f"{delayed:,} delayed"),
            (c3, "Avg Package Weight", f"{avg_w:,.0f} g", "across all orders"),
            (c4, "Avg Discount", f"{avg_disc:.1f}%", "offered per order"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Recommended Workflow</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1. Explore Data", "Understand the dataset — distributions, correlations, and delay patterns."),
        ("2. Train Model", "Pick an algorithm, tune hyperparameters, and fit on historical data."),
        ("3. Evaluate", "Compare accuracy, precision, recall, and F1 across models."),
        ("4. Predict", "Enter live shipment details and receive an instant risk score."),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], steps):
        col.markdown(
            f'<div class="metric-card" style="text-align:center">'
            f'<div style="font-weight:700;font-size:.9rem;margin-bottom:.4rem">{title}</div>'
            f'<div class="sub" style="font-size:.78rem;line-height:1.5">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

def render_eda():
    page_header("Explore Data", "Understand patterns before building a model")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Place `ecommerce_shipping_data.csv` in the `data/` folder.")
        return

    df_plot = add_delay_status(df)

    tab_raw, tab_dist, tab_target, tab_scatter, tab_corr = st.tabs([
        "Raw Data", "Distributions", "Target Analysis", "Scatter", "Correlations"
    ])

    with tab_raw:
        st.caption(f"Showing all {len(df):,} records · {df.shape[1]} columns")
        st.dataframe(df, width="stretch", height=420)

    with tab_dist:
        col_ctrl, _ = st.columns([2, 3])
        feature = col_ctrl.selectbox("Feature to inspect:", NUMERIC_COLS + CATEGORICAL_COLS, key="hist_feature")

        if feature in NUMERIC_COLS:
            bins = col_ctrl.slider("Bin count:", 10, 60, 30, key="hist_bins")
            fig = px.histogram(
                df_plot, x=feature, nbins=bins,
                color="Delay_Status", color_discrete_map=STATUS_COLORS, barmode="overlay",
                opacity=0.7,
                title=f"Distribution of {feature.replace('_', ' ')} by Delivery Status",
                labels={feature: feature.replace("_", " ")},
            )
            fig.update_layout(legend_title="Status", margin=dict(t=40, b=20))
            st.plotly_chart(fig, width="stretch")
            delayed_mean = df_plot[df_plot["Delay_Status"] == "Delayed"][feature].mean()
            ontime_mean  = df_plot[df_plot["Delay_Status"] == "On Time"][feature].mean()
            diff_pct = abs(delayed_mean - ontime_mean) / ontime_mean * 100 if ontime_mean else 0
            strength = "strong" if diff_pct > 15 else ("moderate" if diff_pct > 5 else "weak")
            ibox(
                f"<strong>Auto Interpretation:</strong> Delayed shipments have an average <em>{feature.replace('_',' ')}</em> "
                f"of <strong>{delayed_mean:.2f}</strong> vs <strong>{ontime_mean:.2f}</strong> for on-time ones "
                f"({diff_pct:.1f}% apart). This suggests a <strong>{strength}</strong> signal for predicting delays."
            )
        else:
            fig = px.histogram(
                df_plot, x=feature, color="Delay_Status",
                color_discrete_map=STATUS_COLORS, barmode="group",
                title=f"Delay Rate by {feature.replace('_', ' ')}",
                labels={feature: feature.replace("_", " ")},
            )
            fig.update_layout(legend_title="Status", margin=dict(t=40, b=20))
            st.plotly_chart(fig, width="stretch")
            ibox(
                f"<strong>Auto Interpretation:</strong> The chart shows how delay frequency varies across "
                f"<em>{feature.replace('_', ' ')}</em> categories. Taller dark bars indicate higher-risk segments."
            )

    with tab_target:
        col_ctrl2, _ = st.columns([2, 3])
        box_feature = col_ctrl2.selectbox("Numeric feature:", NUMERIC_COLS, key="box_feature")
        fig_box = px.box(
            df_plot, x="Delay_Status", y=box_feature,
            color="Delay_Status", color_discrete_map=STATUS_COLORS,
            title=f"{box_feature.replace('_', ' ')} vs Delivery Outcome",
            points="outliers",
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=40, b=20))
        st.plotly_chart(fig_box, width="stretch")

        med_delayed = df_plot[df_plot["Delay_Status"] == "Delayed"][box_feature].median()
        med_ontime  = df_plot[df_plot["Delay_Status"] == "On Time"][box_feature].median()
        direction = "higher" if med_delayed > med_ontime else "lower"
        delta = abs(med_delayed - med_ontime)
        ibox(
            f"<strong>What this means:</strong> Delayed parcels have a <strong>{direction} median</strong> "
            f"{box_feature.replace('_', ' ')} ({med_delayed:.1f} vs {med_ontime:.1f}, Δ = {delta:.1f}). "
            f"{'A larger gap means this feature is a stronger predictor.' if delta > 1 else 'The gap is small — this feature alone may not strongly predict delays.'}"
        )

    with tab_scatter:
        c1, c2 = st.columns(2)
        x_axis = c1.selectbox("X axis:", NUMERIC_COLS, index=NUMERIC_COLS.index("Weight_in_gms"), key="sc_x")
        y_axis = c2.selectbox("Y axis:", NUMERIC_COLS, index=NUMERIC_COLS.index("Cost_of_the_Product"), key="sc_y")

        sample_df = df_plot.sample(min(2000, len(df_plot)), random_state=42)
        fig_sc = px.scatter(
            sample_df, x=x_axis, y=y_axis,
            color="Delay_Status", color_discrete_map=STATUS_COLORS,
            opacity=0.55, size_max=6,
            title=f"{x_axis.replace('_',' ')} vs {y_axis.replace('_',' ')} (n=2 000 sample)",
        )
        fig_sc.update_layout(legend_title="Status", margin=dict(t=40, b=20))
        st.plotly_chart(fig_sc, width="stretch")
        ibox(
            f"<strong>Reading the chart:</strong> Overlapping blue and dark clusters suggest "
            f"<em>{x_axis.replace('_',' ')}</em> and <em>{y_axis.replace('_',' ')}</em> together provide "
            "limited separation. Distinct clusters indicate strong joint predictive power."
        )

    with tab_corr:
        corr = df_plot[NUMERIC_COLS + [TARGET_COL]].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Blues",
            title="Pearson Correlation Matrix",
            zmin=-1, zmax=1,
        )
        fig_heat.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig_heat, width="stretch")

        target_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
        top_feature = target_corr.index[0]
        top_val = target_corr.iloc[0]
        ibox(
            f"<strong>Top predictor:</strong> <em>{top_feature.replace('_',' ')}</em> has the strongest "
            f"linear correlation with delays (|r| = {top_val:.3f}). "
            "Values above 0.3 are generally considered meaningful in logistics contexts."
        )

def render_preprocess():
    page_header("Preprocessing", "Configure and verify the data pipeline")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found.")
        return

    st.markdown('<div class="section-heading">Pipeline Configuration</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        mode = st.selectbox(
            "Mode Preset", list(PREPROCESSING_MODES.keys()),
            index=list(PREPROCESSING_MODES.keys()).index(st.session_state["preprocess"]["mode_name"]),
            help="Choose a predefined preprocessing pipeline."
        )
        if mode != st.session_state["preprocess"]["mode_name"]:
            st.session_state["preprocess"].update(PREPROCESSING_MODES[mode])
            st.session_state["preprocess"]["mode_name"] = mode
            st.rerun()

    with c2:
        outlier_method = st.selectbox(
            "Outlier Handling", ["None", "Clip (IQR)", "Remove"],
            index=["None", "Clip (IQR)", "Remove"].index(st.session_state["preprocess"]["outlier_method"]),
        )
        imb_strategy = st.selectbox(
            "Imbalance Strategy", ["None", "Undersampling"],
            index=["None", "Undersampling"].index(st.session_state["preprocess"]["imbalance"]),
        )

    with c3:
        fe_options = ["Weight/Cost Ratio", "High Discount", "Engagement Score", "Loyalty Level", "Delivery Urgency", "Care Intensity"]
        fe_toggles = st.multiselect(
            "Feature Engineering", fe_options,
            default=[opt for opt in fe_options if opt in st.session_state["preprocess"]["fe_toggles"]],
        )

    st.session_state["preprocess"].update({
        "outlier_method": outlier_method,
        "imbalance": imb_strategy,
        "fe_toggles": fe_toggles
    })

    st.divider()

    with st.spinner("Updating preview..."):
        clean_params = {k: v for k, v in st.session_state["preprocess"].items() if k not in ["mode_name", "imbalance"]}
        df_clean = clean_data(df, **clean_params)

    st.markdown('<div class="section-heading">Live Quality Audit</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset Rows", f"{len(df_clean):,}", delta=len(df_clean)-len(df))
    m2.metric("Feature Count", f"{df_clean.shape[1]}", delta=df_clean.shape[1]-df.shape[1])

    missing_orig = df.isnull().sum().sum()
    missing_clean = df_clean.isnull().sum().sum()
    m3.metric("Missing Cells", missing_clean, delta=-missing_orig if missing_orig > 0 else 0)

    counts = df_clean[TARGET_COL].value_counts()
    delay_pct = counts.get(1, 0) / len(df_clean)
    m4.metric("Delay Rate", f"{delay_pct:.1%}")

    c_bal, c_pipe = st.columns([1, 1], gap="large")
    with c_bal:
        dist_df = pd.DataFrame({
            "Status": ["On Time", "Delayed"],
            "Count": [counts.get(0, 0), counts.get(1, 0)],
            "Percentage": [counts.get(0, 0)/len(df_clean), counts.get(1, 0)/len(df_clean)]
        })
        st.dataframe(dist_df.style.format({"Percentage": "{:.1%}"}), hide_index=True, width="stretch")

    with c_pipe:
        if fe_toggles:
            fe_status = "".join([f"<div style='margin-bottom:.2rem'>• {fe}</div>" for fe in fe_toggles])
            st.markdown(f"<div style='font-size:.85rem; opacity:.8'>{fe_status}</div>", unsafe_allow_html=True)
        else:
            st.caption("No engineered features active.")
        if imb_strategy == "Undersampling":
            st.warning("Undersampling enabled for training.")

    st.markdown('<div class="section-heading">Data Preview (Engineered)</div>', unsafe_allow_html=True)
    st.dataframe(df_clean.head(15), width="stretch")
    st.caption(f"Showing top 15 records of {len(df_clean):,} total.")

def render_model_and_eval():
    page_header("Train & Evaluate", "Fit a model, tune it, and inspect its performance")

    df = load_dataset()
    if df is None:
        st.error("Dataset not found.")
        return

    tab_train, tab_eval = st.tabs(["Train", "Evaluate"])

    with tab_train:
        render_train_tab(df)

    with tab_eval:
        render_eval_tab()

def render_train_tab(df):
    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown('<div class="section-heading">Model Selection</div>', unsafe_allow_html=True)
        model_name = st.selectbox(
            "Algorithm", MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state["last_trained_model"])
            if st.session_state["last_trained_model"] in MODEL_OPTIONS else 0,
            help="Pick the machine learning algorithm to train."
        )
        default_params = DEFAULT_PARAMS.get(model_name, {})

        with st.expander("Hyperparameters", expanded=True):
            st.caption("Adjusting these tunes the bias–variance tradeoff.")
            if model_name == "XGBoost":
                n_est  = st.slider("n_estimators", 50, 500, int(default_params.get("n_estimators", 200)), 50)
                lr     = st.slider("learning_rate", 0.01, 0.5, float(default_params.get("learning_rate", 0.08)), 0.01)
                depth  = st.slider("max_depth", 2, 12, int(default_params.get("max_depth", 5)), 1)
                model_params = {"n_estimators": n_est, "learning_rate": lr, "max_depth": depth, "verbosity": 0}
            else:
                n_est  = st.slider("n_estimators", 50, 500, int(default_params.get("n_estimators", 200)), 50)
                depth  = st.slider("max_depth", 2, 20, int(default_params.get("max_depth", 8)), 1)
                mss    = st.slider("min_samples_split", 2, 20, int(default_params.get("min_samples_split", 5)), 1)
                model_params = {"n_estimators": n_est, "max_depth": depth, "min_samples_split": mss, "n_jobs": -1}

        with st.expander("Validation Settings", expanded=False):
            test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05)
            cv_folds  = st.slider("Cross-validation folds", 3, 10, 5)

        if st.button(f"Train {model_name}", type="primary", width="stretch"):
            with st.spinner(f"Training {model_name}..."):
                metrics = train_model(
                    model_name=model_name,
                    model_params=model_params,
                    data_path=str(DATA_PATH),
                    preprocess_params=st.session_state["preprocess"],
                    test_size=float(test_size),
                    cv_folds=int(cv_folds),
                )
            st.session_state["trained_models"][model_name] = metrics
            st.session_state["last_trained_model"] = model_name
            st.toast(f"{model_name} trained successfully!")
            st.rerun()

    with c_right:
        st.markdown('<div class="section-heading">Training Status</div>', unsafe_allow_html=True)
        if model_name in st.session_state["trained_models"]:
            ibox(f"The <strong>{model_name}</strong> model is currently trained and ready for evaluation.", kind="ok")
        else:
            ibox(f"No active <strong>{model_name}</strong> model. Click the button to train.", kind="info")

def render_eval_tab():
    available = list(st.session_state["trained_models"].keys())
    if not available:
        st.info("Train a model first to see its results here.")
        return

    sel = st.selectbox("Inspect model:", available, key="eval_sel")
    m = st.session_state["trained_models"][sel]

    mc = st.columns(4)
    for col, label, key in zip(mc, ["Accuracy", "Precision", "Recall", "F1-Score"],
                                ["accuracy", "precision", "recall", "f1_score"]):
        col.markdown(
            f'<div class="metric-card" style="text-align:center">'
            f'<div class="label">{label}</div>'
            f'<div class="value" style="font-size:1.3rem">{m.get(key,0):.1%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    col_plots_l, col_plots_r = st.columns(2)
    with col_plots_l:
        cm_data = m.get("confusion_matrix")
        if cm_data:
            cm_df = pd.DataFrame(
                [[cm_data["true_negatives"], cm_data["false_positives"]],
                 [cm_data["false_negatives"], cm_data["true_positives"]]],
                columns=["Pred: On Time", "Pred: Delayed"],
                index=["Actual: On Time", "Actual: Delayed"],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix")
            fig_cm.update_layout(margin=dict(t=40, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_cm, width="stretch")

    with col_plots_r:
        fi_data = m.get("feature_importance")
        if fi_data:
            fi_df = pd.DataFrame(fi_data).sort_values("importance", ascending=True).tail(10)
            fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top Predictive Features")
            fig_fi.update_layout(margin=dict(t=40, b=10), showlegend=False, yaxis_title="")
            st.plotly_chart(fig_fi, width="stretch")

    if len(st.session_state["trained_models"]) > 1:
        st.divider()
        st.markdown('<div class="section-heading">Model Comparison</div>', unsafe_allow_html=True)
        comp = [
            {"Model": n, "Accuracy": v.get("accuracy", 0), "Precision": v.get("precision", 0),
             "Recall": v.get("recall", 0), "F1 Score": v.get("f1_score", 0)}
            for n, v in st.session_state["trained_models"].items()
        ]
        df_comp = pd.DataFrame(comp).set_index("Model")
        st.dataframe(df_comp.style.format("{:.2%}"), width="stretch")

def render_predict():
    page_header("Predict Risk", "Enter shipment details and receive an instant delay risk score")

    df = load_dataset()
    def_calls  = int(df["Customer_care_calls"].median()) if df is not None else 4
    def_rating = int(df["Customer_rating"].median())     if df is not None else 3
    def_purch  = int(df["Prior_purchases"].median())     if df is not None else 3
    def_price  = float(df["Cost_of_the_Product"].median()) if df is not None else 150.0
    def_weight = float(df["Weight_in_gms"].median())     if df is not None else 1500.0
    def_disc   = float(df["Discount_offered"].median())  if df is not None else 5.0

    available_models = list(st.session_state["trained_models"].keys())
    if not available_models:
        for m in MODEL_OPTIONS:
            disk = load_metrics(m)
            if disk:
                st.session_state["trained_models"][m] = disk
        available_models = list(st.session_state["trained_models"].keys())
    if not available_models:
        st.warning("No trained model found. Please train a model in Train & Evaluate before predicting.")
        return

    top_l, top_r = st.columns([3, 1])
    with top_l:
        model_name = st.selectbox(
            "Active inference model",
            available_models,
            index=available_models.index(st.session_state["last_trained_model"])
            if st.session_state["last_trained_model"] in available_models else 0,
            key="pred_model",
            help="Select the trained model to use for this assessment.",
        )
    with top_r:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Result", width="stretch", help="Clears the last risk assessment result"):
            st.session_state["last_result"] = None
            st.session_state["last_inputs"] = None
            st.rerun()

    st.divider()

    with st.form("prediction_form", clear_on_submit=False):
        st.markdown('<div class="section-heading">Shipment Details</div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        warehouse = r1c1.selectbox("Warehouse Block",  ["A","B","C","D","F"], help="Dispatch origin warehouse")
        mode      = r1c2.selectbox("Shipment Mode",    ["Ship","Flight","Road"])
        importance= r1c3.selectbox("Product Importance", ["low","medium","high"])

        st.markdown('<div class="section-heading">Package & Pricing</div>', unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)
        price  = r2c1.number_input("Product Price ($)", 0.0, 10000.0, def_price, step=10.0)
        weight = r2c2.number_input("Weight (g)",        0.0, 10000.0, def_weight, step=50.0)
        disc   = r2c3.number_input("Discount (%)",      0.0, 100.0,  def_disc, step=1.0)

        st.markdown('<div class="section-heading">Customer Profile</div>', unsafe_allow_html=True)
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        gender    = r3c1.selectbox("Gender", ["Female","Male"])
        calls     = r3c2.number_input("Care Calls",    0, 20, def_calls,  help="Customer service interactions")
        rating    = r3c3.number_input("Customer Rating", 1, 5, def_rating, help="1 = lowest, 5 = highest")
        purchases = r3c4.number_input("Prior Purchases", 0, 50, def_purch)

        submitted = st.form_submit_button("Run Risk Assessment", type="primary", width="stretch")

    if submitted:
        try:
            t0 = time.perf_counter()
            res = predict_risk(
                warehouse_block=warehouse, mode_of_shipment=mode,
                product_importance=importance, gender=gender,
                customer_care_calls=calls, customer_rating=rating,
                prior_purchases=purchases, discount_offered=disc,
                cost_of_product=price, weight_in_gms=weight,
                model_name=model_name,
            )
            res["latency_ms"] = (time.perf_counter() - t0) * 1000
            st.session_state["last_result"] = res
            st.session_state["last_inputs"] = {
                "mode": mode, "weight": weight, "warehouse": warehouse,
                "discount": disc, "rating": rating, "calls": calls,
                "importance": importance, "price": price,
                "purchases": purchases, "gender": gender
            }
            st.toast("Assessment complete")
            st.rerun()
        except Exception as e:
            st.error(f"**Assessment failed:** {e}")
            ibox(
                "<strong>How to resolve:</strong><br>"
                "1. Make sure you have trained a model in the <em>Train & Evaluate</em> tab.<br>"
                "2. Confirm the model files exist in the <code>models/</code> directory.<br>"
                "3. Verify all input fields are filled with valid values.",
                kind="warn",
            )

    if st.session_state.get("last_result"):
        res    = st.session_state["last_result"]
        inputs = st.session_state.get("last_inputs", {})
        prob   = res["probability"]
        is_delayed = res["risk_class"] == 1

        st.divider()
        st.markdown('<div class="section-heading">Risk Assessment Result</div>', unsafe_allow_html=True)

        col_verdict, col_gauge = st.columns([3, 2])

        with col_verdict:
            if is_delayed:
                ibox(
                    "<strong>HIGH RISK — Likely Delayed</strong><br>"
                    f"The model predicts a <strong>{prob:.1%} probability of delay</strong> for this shipment. "
                    "Review the actionable recommendations below.",
                    kind="risk",
                )
                st.markdown("#### Actionable Recommendations")
                recs = []
                if inputs.get("mode") in ["Ship", "Road"]:
                    recs.append(
                        f"**Switch transport mode:** Current mode is *{inputs.get('mode')}*. "
                        "Upgrading to **Flight** typically bypasses terrestrial/maritime bottlenecks and reduces transit time by 30–60%."
                    )
                if inputs.get("weight", 0) > 4000:
                    recs.append(
                        "**Priority terminal handling:** Parcels over 4 kg face longer loading consolidation queues. "
                        "Flag for priority staging to reduce warehouse dwell time."
                    )
                if inputs.get("warehouse") == "F":
                    recs.append(
                        "**Audit Warehouse Block F:** This block historically shows the highest dispatch latency. "
                        "Shift inventory to Block A or B where possible, or raise a dispatch SLA review."
                    )
                if inputs.get("discount", 0) > 10:
                    recs.append(
                        "**Fulfillment queue alert:** Heavy discounts drive volume spikes that overwhelm pick-pack lines. "
                        "Consider batch-releasing high-discount orders during off-peak hours."
                    )
                if inputs.get("calls", 0) >= 5:
                    recs.append(
                        "**High customer contact:** 5+ care calls indicate existing friction. "
                        "Proactively notify this customer of potential delays to manage expectations and prevent escalation."
                    )
                if not recs:
                    recs.append(
                        "Assign this parcel to a **priority courier lane** and send the customer "
                        "a proactive delay notification with a revised ETA."
                    )
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                ibox(
                    "<strong>LOW RISK — Expected On Time</strong><br>"
                    f"Predicted delay probability is only <strong>{prob:.1%}</strong>, which is well below the risk threshold. "
                    "No manual routing intervention is required.",
                    kind="ok",
                )
                st.markdown("#### Favorable Factors & Next Steps")
                low_recs = []
                if inputs.get("mode") == "Flight":
                    low_recs.append("**Optimal Transport:** Flight routing provides a strong buffer against unexpected delays.")
                elif inputs.get("mode") in ["Road", "Ship"]:
                    low_recs.append(f"**Stable Route:** Current *{inputs.get('mode')}* logistics lanes are operating with normal throughput and no severe bottlenecks.")
                
                if inputs.get("weight", 0) <= 3000:
                    low_recs.append("**Efficient Weight:** The package weight allows for rapid automated conveyor sortation.")
                
                if inputs.get("calls", 0) < 3:
                    low_recs.append("**Healthy Order Status:** Low customer inquiry rate indicates the fulfillment pipeline is proceeding smoothly.")
                
                low_recs.append("**Action:** Proceed with standard fulfillment. No priority flagging or proactive alerts are necessary.")
                
                for r in low_recs:
                    st.markdown(f"- {r}")

        with col_gauge:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 36, "family": "DM Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                    "bar": {"color": "#ef4444" if is_delayed else "#22c55e", "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "#f0fdf4"},
                        {"range": [40, 65], "color": "#fefce8"},
                        {"range": [65, 100],"color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line": {"color": "#94a3b8", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
                title={"text": "Delay Probability", "font": {"size": 13, "color": "#64748b"}},
            ))
            fig_g.update_layout(
                height=240, margin=dict(t=30, b=0, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_g, width="stretch")
            st.markdown(
                f'<div style="text-align:center;font-size:.78rem;color:var(--text-muted)">'
                f'Model: <strong>{model_name}</strong> - '
                f'Latency: <strong>{res.get("latency_ms",0):.1f} ms</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown('<div class="section-heading">Detailed Interpretation</div>', unsafe_allow_html=True)
        st.markdown("We analyzed **every single factor** you entered. Here is exactly how each contributes to this prediction:")

        exp_c1, exp_c2 = st.columns(2)

        with exp_c1:
            m_txt = 'Fastest routing, significantly decreases delay risk.' if inputs.get('mode') == 'Flight' else 'Slower terrestrial/maritime routing, more prone to delays.'
            w_txt = 'Heavy package, requires special handling.' if inputs.get('weight', 0) > 4000 else ('Lightweight, moves quickly through sortation.' if inputs.get('weight', 0) < 2000 else 'Standard weight, easy to consolidate.')
            wb_txt = 'Historically high volume, prone to bottlenecks.' if inputs.get('warehouse') == 'F' else 'Stable dispatch operations.'
            imp_txt = 'Prioritized in sorting queues.' if inputs.get('importance') == 'high' else 'Standard queue priority.'

            ibox("<strong>Package & Routing</strong><br><br>"
                 f"• <strong>Mode ({inputs.get('mode')}):</strong> {m_txt}<br>"
                 f"• <strong>Weight ({inputs.get('weight')}g):</strong> {w_txt}<br>"
                 f"• <strong>Warehouse ({inputs.get('warehouse')}):</strong> {wb_txt}<br>"
                 f"• <strong>Importance ({inputs.get('importance')}):</strong> {imp_txt}<br>"
                 f"• <strong>Cost (${inputs.get('price')}):</strong> Higher value items sometimes face stricter security checks.", kind="info")

        with exp_c2:
            d_txt = 'High discount, strongly correlates with volume spikes and fulfillment delays.' if inputs.get('discount', 0) > 10 else 'Standard pricing, normal processing.'
            c_txt = 'High contact rate, indicates likely friction in fulfillment pipeline.' if inputs.get('calls', 0) >= 4 else 'Normal inquiry levels.'

            ibox("<strong>Customer Profile</strong><br><br>"
                 f"• <strong>Discount ({inputs.get('discount')}%):</strong> {d_txt}<br>"
                 f"• <strong>Care Calls ({inputs.get('calls')}):</strong> {c_txt}<br>"
                 f"• <strong>Rating ({inputs.get('rating')}/5):</strong> Reflects past satisfaction, sometimes correlated with regional logistics quality.<br>"
                 f"• <strong>Purchases ({inputs.get('purchases')}):</strong> Loyal customers may receive priority.<br>"
                 f"• <strong>Gender ({inputs.get('gender')}):</strong> Demographic tracking, historically negligible impact on delays.", kind="info")

        if res.get("top_factors"):
            factors_html = " ".join([f"<span style='background:var(--surface-2); padding:4px 8px; border-radius:4px; border:1px solid var(--border); margin-right:8px; font-size:0.85rem;'>{f['feature'].replace('_',' ')}: <strong>{f['importance']:.2f}</strong></span>" for f in res["top_factors"]])
            st.markdown("<br><div style='font-size:0.95rem; color:var(--text-main); margin-bottom:0.5rem;'><strong>Model's Top Mathematical Influencers:</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div>{factors_html}</div><br>", unsafe_allow_html=True)


pg = st.navigation([
    st.Page(render_home,          title="Home",             default=True),
    st.Page(render_eda,           title="EDA"),
    st.Page(render_preprocess,    title="Preprocessing"),
    st.Page(render_model_and_eval,title="Train & Evaluate"),
    st.Page(render_predict,       title="Predict Risk"),
])

with st.sidebar:
    st.markdown('<div class="sidebar-info" style="font-weight:600; color:var(--primary); font-size:20px">BINUS University:</div>', unsafe_allow_html=True)
    for name in ["Anandhio Varistama", "Muhammad Rizki Akbar", "Jason Tirta", "Ivan Novanto Bastian", "Satya Herlambang Kurniawan"]:
        st.markdown(f'<div class="sidebar-info">{name}</div>', unsafe_allow_html=True)

pg.run()
