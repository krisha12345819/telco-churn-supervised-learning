import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SIGNAL — Churn Risk Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# Scaler statistics computed from the training data (mean / population std)
SCALE_STATS = {
    "tenure":         {"mean": 32.371149, "std": 24.557737},
    "MonthlyCharges": {"mean": 64.761692, "std": 30.087911},
    "TotalCharges":   {"mean": 2283.300441, "std": 2266.610181},
    "num_services":   {"mean": 2.037910,  "std": 1.847551},
}

MODEL_COLUMNS = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'tenure_group',
    'num_services', 'AutoPay', 'InternetService_Fiber optic',
    'InternetService_No', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'gender_Male',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'tenure_bucket_13-24', 'tenure_bucket_25-48', 'tenure_bucket_49-72'
]


def engineer_features(inputs: dict) -> pd.DataFrame:
    """Reproduce the exact feature engineering pipeline used in training."""
    row = {}

    row["SeniorCitizen"] = 1 if inputs["senior"] == "Yes" else 0
    row["Partner"] = 1 if inputs["partner"] == "Yes" else 0
    row["Dependents"] = 1 if inputs["dependents"] == "Yes" else 0
    row["tenure"] = inputs["tenure"]
    row["PhoneService"] = 1 if inputs["phone_service"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if inputs["paperless"] == "Yes" else 0
    row["MonthlyCharges"] = inputs["monthly_charges"]
    row["TotalCharges"] = inputs["total_charges"]

    # tenure_group: ["New","Mid","Senior","Loyal"] -> bins [0,12,36,60,inf]
    t = inputs["tenure"]
    if t <= 12:
        tg = 0
    elif t <= 36:
        tg = 1
    elif t <= 60:
        tg = 2
    else:
        tg = 3
    row["tenure_group"] = tg

    service_flags = [
        inputs["online_security"], inputs["online_backup"],
        inputs["device_protection"], inputs["tech_support"],
        inputs["streaming_tv"], inputs["streaming_movies"]
    ]
    row["num_services"] = sum(1 for s in service_flags if s == "Yes")

    row["AutoPay"] = 1 if "automatic" in inputs["payment_method"].lower() else 0

    # One-hot: InternetService (drop_first -> dropped 'DSL')
    row["InternetService_Fiber optic"] = 1 if inputs["internet_service"] == "Fiber optic" else 0
    row["InternetService_No"] = 1 if inputs["internet_service"] == "No" else 0

    # One-hot: Contract (drop_first -> dropped 'Month-to-month')
    row["Contract_One year"] = 1 if inputs["contract"] == "One year" else 0
    row["Contract_Two year"] = 1 if inputs["contract"] == "Two year" else 0

    # One-hot: PaymentMethod (drop_first -> dropped 'Bank transfer (automatic)')
    row["PaymentMethod_Credit card (automatic)"] = 1 if inputs["payment_method"] == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"] = 1 if inputs["payment_method"] == "Electronic check" else 0
    row["PaymentMethod_Mailed check"] = 1 if inputs["payment_method"] == "Mailed check" else 0

    # gender (drop_first -> dropped 'Female')
    row["gender_Male"] = 1 if inputs["gender"] == "Male" else 0

    # MultipleLines (drop_first -> dropped 'No')
    row["MultipleLines_No phone service"] = 1 if inputs["multiple_lines"] == "No phone service" else 0
    row["MultipleLines_Yes"] = 1 if inputs["multiple_lines"] == "Yes" else 0

    # service add-ons (drop_first -> dropped 'No')
    def addon_cols(prefix, value):
        return {
            f"{prefix}_No internet service": 1 if value == "No internet service" else 0,
            f"{prefix}_Yes": 1 if value == "Yes" else 0,
        }

    row.update(addon_cols("OnlineSecurity", inputs["online_security"]))
    row.update(addon_cols("OnlineBackup", inputs["online_backup"]))
    row.update(addon_cols("DeviceProtection", inputs["device_protection"]))
    row.update(addon_cols("TechSupport", inputs["tech_support"]))
    row.update(addon_cols("StreamingTV", inputs["streaming_tv"]))
    row.update(addon_cols("StreamingMovies", inputs["streaming_movies"]))

    # tenure_bucket (drop_first -> dropped '0-12')
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12", "13-24", "25-48", "49-72"]
    tb = pd.cut([t], bins=bins, labels=labels, include_lowest=True)[0]
    row["tenure_bucket_13-24"] = 1 if tb == "13-24" else 0
    row["tenure_bucket_25-48"] = 1 if tb == "25-48" else 0
    row["tenure_bucket_49-72"] = 1 if tb == "49-72" else 0

    df = pd.DataFrame([row])

    # scale the 4 continuous columns using stored training stats
    for col, stats in SCALE_STATS.items():
        df[col] = (df[col] - stats["mean"]) / stats["std"]

    return df[MODEL_COLUMNS]


def predict(inputs: dict):
    X = engineer_features(inputs)
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return pred, proba, X


# ----------------------------------------------------------------------------
# STYLE — "SIGNAL" command-console theme
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root{
    --bg:        #0a0e14;
    --surface:   #11161f;
    --surface-2: #161d29;
    --line:      #232b38;
    --cyan:      #4fd8c7;
    --amber:     #ffb454;
    --red:       #ff5d6c;
    --text:      #e7ecf2;
    --muted:     #7a8699;
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp{
    background:
        radial-gradient(circle at 12% -10%, rgba(79,216,199,0.07), transparent 40%),
        radial-gradient(circle at 90% 0%, rgba(255,180,84,0.05), transparent 35%),
        var(--bg);
    color: var(--text);
}
#MainMenu, footer, header {visibility:hidden;}
section[data-testid="stSidebar"]{
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container{ padding-top: 1.5rem; }

h1,h2,h3,h4 { font-family:'Space Grotesk', sans-serif !important; letter-spacing:-0.01em; }

.signal-eyebrow{
    font-family:'JetBrains Mono', monospace;
    font-size:0.72rem;
    letter-spacing:0.22em;
    color: var(--cyan);
    text-transform:uppercase;
    margin-bottom:0.3rem;
}
.signal-title{
    font-size:2.4rem;
    font-weight:700;
    color: var(--text);
    margin:0 0 0.25rem 0;
    line-height:1.1;
}
.signal-sub{
    color: var(--muted);
    font-size:0.95rem;
    margin-bottom:1.6rem;
    max-width: 640px;
}

.card{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
}
.card-tight{ padding: 1rem 1.2rem; }

.metric-label{
    font-family:'JetBrains Mono', monospace;
    font-size:0.7rem;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color: var(--muted);
}
.metric-value{
    font-size:1.7rem;
    font-weight:700;
    font-family:'Space Grotesk', sans-serif;
}

.bars{ display:flex; align-items:flex-end; gap:4px; height:26px; }
.bar{ width:6px; border-radius:2px; background: var(--line); }
.bar.on{ background: var(--cyan); }
.bar.on.danger{ background: var(--red); }

.divider{ height:1px; background: var(--line); margin: 1.1rem 0; border:none; }

.tag{
    display:inline-block;
    font-family:'JetBrains Mono', monospace;
    font-size:0.68rem;
    letter-spacing:0.08em;
    padding: 3px 9px;
    border-radius: 999px;
    border:1px solid var(--line);
    color: var(--muted);
    margin-right:6px;
}

.factor-row{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:8px 0;
    border-bottom: 1px dashed var(--line);
    font-size:0.88rem;
}
.factor-row:last-child{ border-bottom:none; }
.factor-name{ color: var(--text); }
.factor-impact-up{ color: var(--red); font-family:'JetBrains Mono', monospace; font-size:0.78rem;}
.factor-impact-down{ color: var(--cyan); font-family:'JetBrains Mono', monospace; font-size:0.78rem;}

div[data-testid="stForm"]{
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.3rem 1.4rem 0.6rem 1.4rem;
}

.stButton>button, .stFormSubmitButton>button{
    background: linear-gradient(135deg, var(--cyan), #2fb8a6);
    color: #04130f;
    font-weight:600;
    border:none;
    border-radius:10px;
    padding: 0.55rem 1.2rem;
    font-family:'Space Grotesk', sans-serif;
    letter-spacing:0.01em;
}
.stButton>button:hover, .stFormSubmitButton>button:hover{
    filter: brightness(1.08);
}

[data-testid="stMetricValue"]{ font-family:'Space Grotesk', sans-serif; }

hr{ border-color: var(--line); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR — Customer profile input
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="signal-eyebrow">Customer Intake</div>', unsafe_allow_html=True)
    st.markdown("### Build a profile")
    st.caption("Enter the account details below, then run the scan.")

    with st.form("profile_form"):
        st.markdown("**Demographics**")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        c3, c4 = st.columns(2)
        partner = c3.selectbox("Has partner", ["No", "Yes"])
        dependents = c4.selectbox("Has dependents", ["No", "Yes"])

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Charges**")
        c5, c6 = st.columns(2)
        monthly_charges = c5.number_input("Monthly ($)", 0.0, 200.0, 70.0, step=0.5)
        total_charges = c6.number_input("Total ($)", 0.0, 10000.0, float(round(monthly_charges * max(tenure, 1), 2)), step=1.0)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

        if internet_service == "No":
            online_security = online_backup = device_protection = "No internet service"
            tech_support = streaming_tv = streaming_movies = "No internet service"
            st.caption("Add-ons disabled — no internet service selected.")
        else:
            c7, c8 = st.columns(2)
            online_security = c7.selectbox("Online security", ["No", "Yes"])
            online_backup = c8.selectbox("Online backup", ["No", "Yes"])
            c9, c10 = st.columns(2)
            device_protection = c9.selectbox("Device protection", ["No", "Yes"])
            tech_support = c10.selectbox("Tech support", ["No", "Yes"])
            c11, c12 = st.columns(2)
            streaming_tv = c11.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = c12.selectbox("Streaming movies", ["No", "Yes"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("▶  Run churn scan", use_container_width=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.markdown('<div class="signal-eyebrow">Telco Retention Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="signal-title">📡 SIGNAL</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="signal-sub">A live read on whether a customer is about to drop off the network — '
        'powered by a decision-tree model trained on 7,043 telecom accounts.</div>',
        unsafe_allow_html=True
    )
with right:
    st.markdown('<div class="tag">DECISION TREE</div><div class="tag">MAX DEPTH 8</div><div class="tag">BALANCED</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------
if not submitted:
    info_l, info_r = st.columns([0.55, 0.45])
    with info_l:
        st.markdown("""
        <div class="card">
        <div class="metric-label">Standing by</div>
        <h3 style="margin-top:0.4rem;">No scan run yet</h3>
        <p style="color:var(--muted); font-size:0.92rem;">
        Fill in the customer profile on the left and hit <b style="color:var(--cyan)">Run churn scan</b> to
        generate a live risk reading, signal-strength gauge, and the factors driving the prediction.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with info_r:
        st.markdown("""
        <div class="card">
        <div class="metric-label">How it reads the signal</div>
        <p style="color:var(--muted); font-size:0.88rem; margin-top:0.5rem;">
        Contract length, tenure, and number of subscribed add-on services tend to weigh most heavily.
        Month-to-month, low-tenure, low-attachment accounts run hottest.
        </p>
        </div>
        """, unsafe_allow_html=True)
else:
    inputs = dict(
        gender=gender, senior=senior, partner=partner, dependents=dependents,
        tenure=tenure, contract=contract, paperless=paperless, payment_method=payment_method,
        monthly_charges=monthly_charges, total_charges=total_charges,
        phone_service=phone_service, multiple_lines=multiple_lines,
        internet_service=internet_service, online_security=online_security,
        online_backup=online_backup, device_protection=device_protection,
        tech_support=tech_support, streaming_tv=streaming_tv, streaming_movies=streaming_movies,
    )

    pred, proba, X = predict(inputs)
    risk_pct = proba * 100

    if risk_pct >= 65:
        risk_label, risk_color = "HIGH RISK", "#ff5d6c"
    elif risk_pct >= 35:
        risk_label, risk_color = "MODERATE RISK", "#ffb454"
    else:
        risk_label, risk_color = "LOW RISK", "#4fd8c7"

    col_gauge, col_detail = st.columns([0.42, 0.58])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 46, "color": "#e7ecf2", "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#7a8699", "tickfont": {"color": "#7a8699"}},
                "bar": {"color": risk_color, "thickness": 0.28},
                "bgcolor": "#161d29",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(79,216,199,0.12)"},
                    {"range": [35, 65], "color": "rgba(255,180,84,0.12)"},
                    {"range": [65, 100], "color": "rgba(255,93,108,0.12)"},
                ],
                "threshold": {"line": {"color": "#e7ecf2", "width": 2}, "thickness": 0.75, "value": risk_pct},
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e7ecf2"},
            margin=dict(t=10, b=10, l=20, r=20),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        bars_html = ""
        n_on = int(round(risk_pct / 10))
        for i in range(10):
            cls = "bar on danger" if (i < n_on and risk_pct >= 65) else ("bar on" if i < n_on else "bar")
            height = 8 + i * 2
            bars_html += f'<div class="{cls}" style="height:{height}px;"></div>'

        st.markdown(f"""
        <div class="card card-tight" style="text-align:center;">
            <div class="metric-label">Churn signal strength</div>
            <div class="bars" style="justify-content:center; margin:10px 0 6px;">{bars_html}</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; color:{risk_color}; letter-spacing:0.05em;">
                {risk_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        verdict = "Likely to churn" if pred == 1 else "Likely to stay"
        verdict_color = "#ff5d6c" if pred == 1 else "#4fd8c7"
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Model verdict</div>
            <div class="metric-value" style="color:{verdict_color};">{verdict}</div>
            <p style="color:var(--muted); font-size:0.88rem; margin-top:0.6rem;">
                Estimated churn probability of <b style="color:{risk_color}">{risk_pct:.1f}%</b> based on the
                profile submitted. This account is on a <b>{contract}</b> contract with
                <b>{tenure} months</b> of tenure and <b>{X['num_services'].iloc[0] if False else sum(1 for s in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies] if s=='Yes')}</b> active add-on services.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Heuristic, human-readable risk factors (directional, based on known churn drivers)
        factors = []
        factors.append(("Contract type", contract,
                         "up" if contract == "Month-to-month" else "down"))
        factors.append(("Tenure", f"{tenure} mo",
                         "up" if tenure <= 12 else "down"))
        n_services = sum(1 for s in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies] if s == "Yes")
        factors.append(("Add-on services", f"{n_services}/6 active",
                         "up" if n_services <= 1 else "down"))
        factors.append(("Internet service", internet_service,
                         "up" if internet_service == "Fiber optic" else "down"))
        factors.append(("Payment method", payment_method,
                         "down" if "automatic" in payment_method.lower() else "up"))
        factors.append(("Monthly charges", f"${monthly_charges:.2f}",
                         "up" if monthly_charges >= 70 else "down"))

        rows = ""
        for name, val, direction in factors:
            cls = "factor-impact-up" if direction == "up" else "factor-impact-down"
            arrow = "▲ raises risk" if direction == "up" else "▼ lowers risk"
            rows += f'<div class="factor-row"><span class="factor-name">{name} — <span style="color:var(--muted)">{val}</span></span><span class="{cls}">{arrow}</span></div>'

        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Key factors in this read</div>
            <div style="margin-top:0.5rem;">{rows}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("🔍  View engineered feature vector sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

st.markdown(
    '<p style="text-align:center; color:var(--muted); font-size:0.78rem; margin-top:2rem;">'
    'SIGNAL · Telco Customer Churn Console · Decision Tree model trained on the IBM Telco Customer Churn dataset'
    '</p>', unsafe_allow_html=True
)
