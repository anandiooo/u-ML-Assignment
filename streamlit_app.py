import streamlit as st
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set global plot style
sns.set_theme(style="whitegrid", palette="Blues")

# add src folder to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import predict_risk, get_model_and_featurizer

# file paths
DATA_PATH = Path("data/ecommerce_shipping_data.csv")
METRICS_PATH = Path("models/metrics.json")
MODEL_PATH = Path("models/xgboost_model.pkl")
FEATURE_NAMES_PATH = Path("models/features.json")
TARGET_COL = "Reached.on.Time_Y.N"
NUMERIC_COLS = [
    "Customer_care_calls",
    "Customer_rating",
    "Cost_of_the_Product",
    "Prior_purchases",
    "Discount_offered",
    "Weight_in_gms",
]
CATEGORICAL_COLS = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Product_importance",
    "Gender",
]

# plot helpers
def create_fig(aspect="wide"):
    # create a figure with a consistent aspect ratio
    sizes = {"wide": (6, 3.6), "square": (4, 3)}
    fig, ax = plt.subplots(figsize=sizes.get(aspect, sizes["wide"]))
    return fig, ax

def show_fig(fig):
    # render and close a matplotlib figure in streamlit
    st.pyplot(fig, use_container_width=True, transparent=False, facecolor="white")
    plt.close(fig)

# data helpers
@st.cache_data
def load_dataset():
    # load dataset from disk
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    # load metrics from disk
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def cached_load_model():
    # load model and featurizer into memory
    try:
        return get_model_and_featurizer()
    except FileNotFoundError:
        return None, None


def add_delay_status(df):
    df_plot = df.copy()
    df_plot["Delay_Status"] = df_plot[TARGET_COL].map({0: "On Time", 1: "Delayed"})
    return df_plot


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# page renderers
def render_home():
    st.title("LogiRisk: Delivery Delay Risk Prediction System")
    st.markdown("An intelligent dashboard that evaluates historical shipment data to predict whether a package will arrive on time.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Problem")
        st.markdown(
            "Delivery delays cause operational inefficiency, higher logistics costs, and lower customer satisfaction. "
            "Without proactive prediction, routing and courier prioritization often rely on intuition rather than data."
        )

    with col2:
        st.subheader("The Solution")
        st.markdown(
            "LogiRisk uses an advanced machine learning model to analyze cargo variables in real-time. "
            "By identifying at-risk shipments before dispatch, logistics teams can make proactive, data-driven decisions."
        )

    st.markdown("---")

    st.subheader("How It Works")
    col_step1, col_step2, col_step3 = st.columns(3)
    with col_step1:
        st.info("**1. Input Details**  \nEnter shipment parameters like weight, product price, and customer history.")
    with col_step2:
        st.info("**2. Model Predicts**  \nThe system evaluates the data against historical delay patterns.")
    with col_step3:
        st.info("**3. View Result**  \nThe dashboard instantly returns a risk assessment and probability score.")

    st.markdown("---")

    st.subheader("Key Output")
    st.markdown(
        "- **Low Risk (On Time):** The package is likely to reach the customer on schedule. No intervention needed.\n"
        "- **High Risk (Delayed):** The package is at risk of delay. Consider prioritizing or changing the shipment mode.\n"
        "- **Probability Score:** A percentage indicating the likelihood of the delay."
    )


def render_dashboard(df):
    if df is None:
        st.warning("Dataset not found. Please place `ecommerce_shipping_data.csv` in the `data/` folder.")
        return

    total = len(df)
    delayed = int(df[TARGET_COL].sum())
    on_time = total - delayed
    delay_rate = delayed / total if total else 0

    avg_cost = df["Cost_of_the_Product"].mean()
    median_cost = df["Cost_of_the_Product"].median()
    avg_weight = df["Weight_in_gms"].mean()
    median_weight = df["Weight_in_gms"].median()
    avg_discount = df["Discount_offered"].mean()
    avg_rating = df["Customer_rating"].mean()

    top_warehouse = df["Warehouse_block"].value_counts().idxmax()
    top_mode = df["Mode_of_Shipment"].value_counts().idxmax()

    row1 = st.columns(4)
    row1[0].metric("Total Shipments", f"{total:,}")
    row1[1].metric("Delay Rate", f"{delay_rate:.2%}")
    row1[2].metric("Avg Product Price", f"${avg_cost:,.2f}")
    row1[3].metric("Avg Weight (g)", f"{avg_weight:,.0f}")

    row2 = st.columns(4)
    row2[0].metric("Median Price", f"${median_cost:,.2f}")
    row2[1].metric("Median Weight (g)", f"{median_weight:,.0f}")
    row2[2].metric("Avg Discount", f"{avg_discount:,.1f}")
    row2[3].metric("Avg Customer Rating", f"{avg_rating:.2f}")

    row3 = st.columns(2)
    row3[0].metric("Top Warehouse", top_warehouse)
    row3[1].metric("Top Shipment Mode", top_mode)


def render_prediction(df=None):
    st.header("Predict Shipment Risk")
    st.markdown("Enter shipment details below to evaluate the risk of delay.")

    if df is not None and {"Cost_of_the_Product", "Weight_in_gms"}.issubset(df.columns):
        price_p5, price_p95 = df["Cost_of_the_Product"].quantile([0.05, 0.95])
        weight_p5, weight_p95 = df["Weight_in_gms"].quantile([0.05, 0.95])
        st.caption(
            "Typical ranges (5th-95th percentile): "
            f"Price ${price_p5:,.0f}-${price_p95:,.0f}, "
            f"Weight {weight_p5:,.0f}-{weight_p95:,.0f} g."
        )
    else:
        st.caption("Tip: Use values within the dataset range for best accuracy.")

    with st.form("prediction_form"):
        st.markdown("**Shipment Details**")
        col1, col2 = st.columns(2)
        with col1:
            warehouse_block = st.selectbox(
                "Warehouse Block",
                ["A", "B", "C", "D", "F"],
                help="The block where the package is stored."
            )
            mode_of_shipment = st.selectbox(
                "Shipment Mode",
                ["Ship", "Flight", "Road"],
                help="The method of transportation."
            )
        with col2:
            product_importance = st.selectbox(
                "Product Importance",
                ["low", "medium", "high"],
                help="Priority level of the product."
            )
            gender = st.selectbox(
                "Customer Gender",
                ["F", "M"],
                help="Gender of the customer."
            )

        st.markdown("**Customer History & Interaction**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            customer_care_calls = st.number_input(
                "Customer Care Calls",
                min_value=0, max_value=20, value=4, step=1,
                help="Number of calls made by the customer regarding the shipment."
            )
        with col_c2:
            customer_rating = st.number_input(
                "Customer Rating",
                min_value=1, max_value=5, value=3, step=1,
                help="Customer's rating of previous service (1-5)."
            )
        with col_c3:
            prior_purchases = st.number_input(
                "Prior Purchases",
                min_value=0, max_value=50, value=3, step=1,
                help="Number of prior purchases made by this customer."
            )

        st.markdown("**Product Attributes**")
        col3, col4, col5 = st.columns(3)
        with col3:
            cost_of_product = st.number_input(
                "Product Price ($)",
                min_value=0.0,
                value=150.0,
                step=10.0,
                format="%.2f",
                help="The declared value or price of the product."
            )
        with col4:
            weight_in_gms = st.number_input(
                "Weight (grams)",
                min_value=0.0,
                value=1500.0,
                step=50.0,
                format="%.2f",
                help="The total weight of the package."
            )
        with col5:
            discount_offered = st.number_input(
                "Discount Offered (%)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                format="%.2f",
                help="Discount offered on the product."
            )

        submitted = st.form_submit_button("Predict Risk", type="primary")

    if submitted:
        if cost_of_product <= 0 or weight_in_gms <= 0:
            st.warning("Please ensure that both Product Price and Weight are greater than 0.")
        else:
            try:
                import time
                with st.spinner("Analyzing risk..."):
                    start = time.perf_counter()
                    result = predict_risk(
                        warehouse_block=warehouse_block,
                        mode_of_shipment=mode_of_shipment,
                        product_importance=product_importance,
                        gender=gender,
                        customer_care_calls=customer_care_calls,
                        customer_rating=customer_rating,
                        prior_purchases=prior_purchases,
                        discount_offered=float(discount_offered),
                        cost_of_product=float(cost_of_product),
                        weight_in_gms=float(weight_in_gms)
                    )
                    latency_ms = (time.perf_counter() - start) * 1000
                    result["latency_ms"] = latency_ms

                st.session_state["last_result"] = result
                st.session_state["last_inputs"] = {
                    "Warehouse": warehouse_block,
                    "Mode": mode_of_shipment,
                    "Importance": product_importance,
                    "Gender": gender,
                    "Calls": customer_care_calls,
                    "Rating": customer_rating,
                    "Prior Purchases": prior_purchases,
                    "Discount (%)": discount_offered,
                    "Price ($)": cost_of_product,
                    "Weight (g)": weight_in_gms
                }
                st.success("Prediction successful. Results are shown below.")
            except FileNotFoundError:
                st.error("Model files not found. Please train the model first by running `python src/train.py`.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")


def render_dataset(df):
    st.header("Dataset Overview")
    st.markdown("A quick look at the raw shipping data.")

    if df is None:
        st.warning("Dataset not found. Please place `ecommerce_shipping_data.csv` in the `data/` folder.")
        return

    col_ds1, col_ds2 = st.columns(2)
    col_ds1.metric("Total Rows", f"{len(df):,}")
    col_ds2.metric("Total Columns", f"{len(df.columns)}")

    st.markdown("**Sample Data**")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Statistical Summary**")
    st.dataframe(df.describe().round(2), use_container_width=True)


def render_eda(df):
    st.header("Exploratory Data Analysis")

    if df is None:
        st.warning("Dataset not found. Please place `ecommerce_shipping_data.csv` in the `data/` folder.")
        return

    df_plot = add_delay_status(df)
    total = len(df_plot)
    delayed = int(df_plot[TARGET_COL].sum())
    on_time = total - delayed
    delay_rate = delayed / total if total else 0

    # create tabs
    tab_hist, tab_box, tab_scatter, tab_heat = st.tabs([
        "Histogram / Distribution",
        "Boxplot (Target Analysis)",
        "Scatterplot",
        "Heatmap"
    ])

    # sample for scatter plots
    sample_df = df_plot if len(df_plot) <= 5000 else df_plot.sample(5000, random_state=42)

    with tab_hist:
        st.subheader("Feature Distributions")
        st.write("Understand how your data is distributed across different categories and ranges.")

        col1, col2 = st.columns([1, 2])
        feature = col1.selectbox("Select Feature:", NUMERIC_COLS + CATEGORICAL_COLS, key="hist_feature")

        if feature in NUMERIC_COLS:
            bins = col2.slider("Number of Bins:", 10, 60, 30, key="hist_bins")

        st.divider()

        if feature in NUMERIC_COLS:
            fig, ax = create_fig("wide")
            sns.histplot(df_plot[feature], kde=True, bins=bins, ax=ax, color="#6baed6")
            ax.set_title(f"Distribution of {feature.replace('_', ' ')}")
            show_fig(fig)
        else:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                fig1, ax1 = create_fig("square")
                sns.countplot(x=feature, data=df_plot, ax=ax1, color="#4292c6")
                ax1.set_title(f"Count of {feature.replace('_', ' ')}")
                plt.xticks(rotation=45)
                show_fig(fig1)
            with col_p2:
                fig2, ax2 = create_fig("square")
                rate_df = df_plot.groupby(feature)[TARGET_COL].mean().reset_index()
                sns.barplot(data=rate_df, x=feature, y=TARGET_COL, ax=ax2, color="#084594")
                ax2.set_title(f"Delay Rate by {feature.replace('_', ' ')}")
                ax2.set_ylim(0, 1)
                plt.xticks(rotation=45)
                show_fig(fig2)

    with tab_box:
        st.subheader("Distribution Analysis by Target")
        st.write("Compare how numeric features vary between packages that arrived on time vs. delayed.")

        col1, _ = st.columns([1, 2])
        box_feature = col1.selectbox("Select Numeric Feature:", NUMERIC_COLS, index=NUMERIC_COLS.index("Weight_in_gms"), key="box_feature")

        st.divider()

        fig, ax = create_fig("wide")
        sns.boxplot(x="Delay_Status", y=box_feature, data=df_plot, ax=ax, palette=["#9ecae1", "#084594"])
        ax.set_title(f"{box_feature.replace('_', ' ')} distribution by Delay Status")
        show_fig(fig)
        st.caption("Boxplots help identify outliers and differences in medians between the two groups.")

    with tab_scatter:
        st.subheader("Relationship Analysis")
        st.write("Explore correlations between two numeric features, colored by whether they were delayed or on time.")

        col1, col2, _ = st.columns([1, 1, 1])
        x_axis = col1.selectbox("X Axis:", NUMERIC_COLS, index=NUMERIC_COLS.index("Weight_in_gms"), key="scat_x")
        y_axis = col2.selectbox("Y Axis:", NUMERIC_COLS, index=NUMERIC_COLS.index("Cost_of_the_Product"), key="scat_y")

        st.divider()

        fig, ax = create_fig("wide")
        sns.scatterplot(
            data=sample_df, x=x_axis, y=y_axis, hue="Delay_Status",
            palette={"On Time": "#9ecae1", "Delayed": "#084594"}, alpha=0.6, ax=ax
        )
        ax.set_title(f"{x_axis.replace('_', ' ')} vs {y_axis.replace('_', ' ')}")
        show_fig(fig)
        st.caption("Showing a sample of up to 5,000 records for performance.")

    with tab_heat:
        st.subheader("Global Correlation Overview")
        st.write("A bird's-eye view of how all numeric variables relate to each other and the target.")

        st.divider()

        corr = df_plot[NUMERIC_COLS + [TARGET_COL]].corr()
        fig, ax = create_fig("wide")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Numeric Feature Correlation Matrix")
        show_fig(fig)
        st.caption("Values closer to 1 or -1 indicate strong positive or negative correlations, respectively.")



def render_preprocessing():
    st.header("Preprocessing Steps")
    st.markdown("How the raw data is transformed before training.")

    st.info(
        "1. Missing values are handled using median for numeric features and mode for categorical features.\n"
        "2. Categorical variables are converted using One-Hot Encoding.\n"
        "3. Numeric features are kept as-is because XGBoost handles unscaled data well."
    )

    if FEATURE_NAMES_PATH.exists():
        st.markdown("**Features after encoding:**")
        with open(FEATURE_NAMES_PATH, "r") as f:
            feature_names = json.load(f)
        st.code("\n".join(feature_names))
    else:
        st.warning("Feature names will appear after training.")


def render_training():
    st.header("Model Training Metrics")
    st.markdown("Performance of the XGBoost classifier on the validation set.")

    metrics = load_metrics()

    if MODEL_PATH.exists() and metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        col_m2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        col_m3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        col_m4.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")

        st.caption("F1-Score measures the balance between precision and recall, crucial here due to dataset noise.")

        st.markdown("---")
        col_charts1, col_charts2 = st.columns(2)

        with col_charts1:
            cm = metrics.get("confusion_matrix")
            if cm:
                st.markdown("**Confusion Matrix**")
                cm_df = pd.DataFrame(
                    [[cm["true_negatives"], cm["false_positives"]],
                     [cm["false_negatives"], cm["true_positives"]]],
                    columns=["Pred On Time", "Pred Delayed"],
                    index=["Actual On Time", "Actual Delayed"]
                )
                fig, ax = create_fig("square")
                sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_ylabel("True Label")
                ax.set_xlabel("Predicted Label")
                show_fig(fig)

        with col_charts2:
            fi = metrics.get("feature_importance")
            if fi:
                st.markdown("**Feature Importance**")
                fi_df = pd.DataFrame(fi).head(8)  # show top 8
                fig, ax = create_fig("square")
                sns.barplot(data=fi_df, x="importance", y="feature", ax=ax, color="#2171b5")
                ax.set_xlabel("Importance Score")
                ax.set_ylabel("")
                show_fig(fig)
    else:
        st.warning("Model not trained yet. Please run `python src/train.py`.")


def render_result():
    st.header("Prediction Result")

    if "last_result" not in st.session_state:
        st.info("No prediction made yet. Please go to the Prediction page and run a prediction.")
        return

    res = st.session_state["last_result"]
    inp = st.session_state["last_inputs"]

    risk_class = res["risk_class"]
    probability = res["probability"]
    confidence = res["confidence"]

    if risk_class == 0:
        st.success("Low Risk (On Time)")
        st.markdown("This package is likely to be delivered on time. No special intervention is needed.")
    else:
        st.error("High Risk (Delayed)")
        st.markdown("This package is at a high risk of delay. Consider prioritizing or changing the shipment mode.")

    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Probability of Delay", f"{probability * 100:.2f}%")
    col_res2.metric("Model Confidence", f"{confidence * 100:.2f}%")
    col_res3.metric("Inference Latency", f"{res.get('latency_ms', 0):.2f} ms")

    st.markdown("**Risk Probability Indicator:**")
    st.progress(float(probability))

    st.markdown("**Top Contributing Factors:**")
    for factor in res.get("top_factors", []):
        st.caption(f"- {factor['feature'].replace('_', ' ')} (importance: {factor['importance']:.3f})")

    st.divider()

    st.markdown("**Input Summary:**")
    st.json(inp)


# app entry point
st.set_page_config(page_title="LogiRisk: Delivery Delay Risk Prediction System", page_icon="📦", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Montserrat', sans-serif;
}

h1, h2, h3, h4 {
  font-family: 'Montserrat', sans-serif;
  letter-spacing: -0.02em;
}

header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True
)

pages = [
    "Home",
    "Dataset",
    "Exploratory Data Analysis",
    "Preprocessing",
    "Model Training",
    "Prediction & Result",
]

df = load_dataset()

with st.sidebar:
    st.divider()

    st.markdown("### Navigation")
    page = st.radio("Pages", pages, key="nav_page", label_visibility="collapsed")

    st.divider()

if page == "Home":
    render_home()
elif page == "Dataset":
    render_dataset(df)
    render_dashboard(df)
elif page == "Exploratory Data Analysis":
    render_eda(df)
elif page == "Preprocessing":
    render_preprocessing()
elif page == "Model Training":
    render_training()
elif page == "Prediction & Result":
    render_prediction(df)
    st.divider()
    render_result()
