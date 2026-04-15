import streamlit as st
import pickle
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobGuard — Fake Job Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── background: deep navy with subtle dot grid ── */
.stApp {
    background-color: #080e1c;
    background-image: radial-gradient(circle, rgba(99,179,237,0.07) 1px, transparent 1px);
    background-size: 28px 28px;
}

/* main block */
.block-container {
    max-width: 840px !important;
    padding-top: 2.5rem !important;
}

/* headings */
h1 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; color: #f0f4ff !important; letter-spacing: -0.03em !important; }
h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #c9d6f0 !important; }
p, li { color: #8a9cc0 !important; }

/* inputs */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea,
.stSelectbox>div>div>div {
    background-color: #0f1a2e !important;
    border: 1px solid #1e2e4a !important;
    border-radius: 8px !important;
    color: #e8eeff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    transition: border-color 0.2s !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.10) !important;
}

/* labels */
.stTextInput label, .stTextArea label, .stSelectbox label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #f59e0b !important;
    font-weight: 500 !important;
}

/* button */
.stButton>button {
    background: linear-gradient(135deg, #1e3a6e 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.04em !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
    border-color: #60a5fa !important;
    transform: translateY(-1px) !important;
}

/* divider */
hr { border-color: #1a2742 !important; }

/* metric */
[data-testid="metric-container"] {
    background: #0c1628;
    border: 1px solid #1e2e4a;
    border-radius: 10px;
    padding: 18px 22px !important;
    border-top: 2px solid #f59e0b !important;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #4a6080 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f4ff !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background-color: #060d1a !important;
    border-right: 1px solid #1a2742 !important;
}
section[data-testid="stSidebar"] * { color: #8a9cc0 !important; }
section[data-testid="stSidebar"] h2 { color: #f0f4ff !important; font-size: 1.1rem !important; }
section[data-testid="stSidebar"] h3 { color: #f59e0b !important; font-size: 0.85rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
section[data-testid="stSidebar"] strong { color: #c9d6f0 !important; }

/* progress bar track */
.stProgress > div > div { background-color: #1a2742 !important; border-radius: 99px !important; }
.stProgress > div > div > div { border-radius: 99px !important; }

/* expander */
.streamlit-expanderHeader {
    background-color: #0c1628 !important;
    border: 1px solid #1e2e4a !important;
    border-radius: 8px !important;
    color: #8a9cc0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ── load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, tfidf, scaler

model, tfidf, scaler = load_artifacts()


# ── helper ────────────────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'&amp;', 'and', text)
    text = re.sub(r'&lt;', '<', text)   # fix: match notebook preprocessing
    text = re.sub(r'&gt;', '>', text)   # fix: match notebook preprocessing
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── rule-based scam detector ─────────────────────────────────────────────────
# Catches scams the ML model misses: short text, stripped currency symbols,
# Indian/regional patterns underrepresented in training data
# ── scoring weights ──────────────────────────────────────────────────────────
# Each rule carries a weight:
#   3 = definitive fraud signal  (payment demand, identity harvesting)
#   2 = strong fraud signal      (urgency, unrealistic earnings, WhatsApp-only)
#   1 = soft warning             (refundable claims, vague qualifications)
# Thresholds:  0-1 → Real  |  2-3 → Suspicious  |  4+ → Fraudulent

SCAM_RULES = [
    # ── hard fraud (weight 3) ─────────────────────────────────────────────
    (r'\bregistration\s*fee\b',                3,
        "Requires upfront payment (registration/assessment fee) — verified employers never charge candidates"),
    (r'\bprocessing\s*fee\b',                  3,
        "Requires upfront payment (processing/admin fee) — no legitimate hiring process charges applicants"),
    (r'\bpay\b.{0,30}(to (join|start|apply|register))\b', 3,
        "Requires payment to begin employment — a defining characteristic of job scams"),
    (r'\bbank\s*account\s*(number|details)\b', 3,
        "Solicits bank account details at application stage — a direct financial fraud indicator"),
    (r'\bpassport\s*(copy|number|details)\b',  3,
        "Requests passport information upfront — legitimate employers collect documents post-offer only"),
    (r'\b(aadhar|aadhaar|pan\s*card)\b',       3,
        "Requests government ID (Aadhar/PAN) before any formal offer — identity theft risk"),
    (r'\btyping\s*(work|job)\b.{0,60}\b(home|online)\b', 3,
        "Online typing/copy-paste roles are among the most frequently reported job scam formats"),
    (r'\beasy (money|income|earning|cash)\b',   3,
        "Uses 'easy money' language — legitimate postings describe work, not shortcuts"),
    (r'\bdata\s*entry.{0,40}\bno\s*experience\b', 3,
        "Data entry with no experience required — commonly used as cover for fee-based scams"),
    (r'\bguaranteed\s*(income|salary|earning|job)\b', 3,
        "Guarantees income or employment — no employer can legally or practically make this promise"),

    # ── strong signals (weight 2) ─────────────────────────────────────────
    (r'\bearn\b.{0,40}\bper (week|month|day)\b', 2,
        "Advertises unrealistic fixed earnings (per week/month/day) with no role-based justification"),
    (r'([\u20b9]|rs\.?|inr)\s*[\d,]+',         2,
        "Mentions specific currency amounts without a defined salary structure or role scope"),
    (r'\$\s*[\d,]+\s*per (week|day)\b',        2,
        "Quotes a fixed payout per day/week — inconsistent with legitimate employment contracts"),
    (r'\blimited\s*seats?\b',                  2,
        "Uses artificial scarcity ('limited seats') to pressure quick decisions without due diligence"),
    (r'\bjoin\s*(today|now|immediately|instantly)\b', 2,
        "Pushes for immediate joining without a standard interview or verification process"),
    (r'\b(urgent(ly)?|immediate)\s*(hiring|opening|vacancy|requirement)\b', 2,
        "Manufactured urgency in hiring — used to bypass candidate scrutiny"),
    (r'\b(work|job)\s*(from|at)\s*home\b.{0,60}\bearn', 2,
        "Combines remote work promise with earnings guarantee — a high-frequency scam pattern"),
    (r'\bwhatsapp\s*(only|number|me|us|contact)\b', 2,
        "Recruitment conducted exclusively via WhatsApp — outside standard professional hiring channels"),
    (r'\b(gmail|yahoo|hotmail)\.com\b',         2,
        "Contact address is a personal email domain — verified companies use official corporate domains"),

    # ── soft warnings (weight 1) ──────────────────────────────────────────
    (r'\bsecurity\s*deposit\b',                1,
        "Mentions a refundable security deposit — while sometimes legitimate, this is a known fraud tactic worth verifying"),
    (r'\brefundable\b.{0,40}(fee|deposit|amount|charge)\b', 1,
        "Claims fee is refundable — a common reassurance tactic used in job scams to lower guard"),
    (r'\bpaid\s*(onboard|training|evaluation|assessment|program)\b', 1,
        "Requires a paid onboarding/training program — may be legitimate but warrants independent verification"),
    (r'\bno\s*(experience|qualification|skill)\s*(needed|required|necessary)\b', 1,
        "Waives all qualifications for a paid role — inconsistent with standard hiring norms"),
    (r'\bfree\s*(visa|accommodation|ticket|flight)\b', 1,
        "Promises free visa or accommodation — common bait in overseas job scams"),
]


def score_rules(title, requirements):
    """Returns (total_score, list_of_triggered_reasons)."""
    combined_raw = (title + ' ' + requirements).lower()
    total  = 0
    hits   = []
    for pattern, weight, reason in SCAM_RULES:
        if re.search(pattern, combined_raw):
            total += weight
            hits.append((weight, reason))
    # sort highest weight first so most critical flags appear at top
    hits.sort(key=lambda x: x[0], reverse=True)
    return total, [r for _, r in hits]


# verdict tiers
REAL       = 0   # score 0-1  AND  ML says real
SUSPICIOUS = 1   # score 2-3  OR  ML uncertain with some hits
FRAUDULENT = 2   # score 4+   OR  hard hit (weight-3 rule) with ML real


def predict(title, requirements, emp_type, industry, function_):
    req_clean  = clean_text(requirements)
    raw_title  = title.strip() if isinstance(title, str) else ''
    combined   = raw_title + ' ' + req_clean

    text_vec   = tfidf.transform([combined])
    char_count = len(req_clean)
    word_count = len(req_clean.split())
    emp_known  = int(emp_type != 'Not Mentioned')
    ind_known  = int(bool(industry.strip()) and industry.strip() != 'Not Mentioned')
    fn_known   = int(function_.strip() not in ('', 'Not Mentioned'))

    extra      = scaler.transform([[char_count, word_count, emp_known, ind_known, fn_known]])
    final_vec  = hstack([text_vec, csr_matrix(extra)])

    ml_pred    = int(model.predict(final_vec)[0])
    proba      = model.predict_proba(final_vec)[0]
    ml_conf    = proba[ml_pred] * 100

    rule_score, rule_hits = score_rules(title, requirements)

    # ── 3-tier decision ───────────────────────────────────────────────────
    hard_hit = any(
        re.search(p, (title + ' ' + requirements).lower())
        for p, w, _ in SCAM_RULES if w == 3
    )

    if rule_score >= 4 or (hard_hit and ml_pred == 0):
        verdict    = FRAUDULENT
        confidence = max(ml_conf, 88.0)
    elif rule_score >= 2 or (rule_hits and ml_pred == 0):
        verdict    = SUSPICIOUS
        confidence = max(ml_conf, 72.0)
    else:
        verdict    = REAL if ml_pred == 0 else FRAUDULENT
        confidence = ml_conf

    return verdict, round(confidence, 1), rule_score, rule_hits


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️  JobGuard")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Trained on **17,880** real-world job postings. "
        "Uses **TF-IDF + Random Forest** to detect fraudulent listings."
    )
    st.markdown("---")
    st.markdown("### Model Stats")
    st.markdown("- 📊 **Accuracy:** 93%")
    st.markdown("- 🎯 **Fake Recall:** 83%")
    st.markdown("- ⚖️ **Balanced class weights**")
    st.markdown("- 🌲 **Model:** Random Forest (100 trees)")
    st.markdown("- 📝 **Features:** TF-IDF (5k) + 5 engineered")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. Your text is cleaned & lowercased\n"
        "2. TF-IDF converts words → numbers\n"
        "3. Extra features (text length, known fields) added\n"
        "4. Random Forest predicts Real or Fake\n"
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b6b85;font-family:monospace'>🛡️ Astra Project &nbsp;·&nbsp; Final Submission</small>",
        unsafe_allow_html=True
    )


# ── header ────────────────────────────────────────────────────────────────────


st.markdown(
    "<h1 style='font-size:3rem;margin-bottom:6px'>Job<span style='color:#00e5a0'>Guard</span></h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#8a9cc0;margin-bottom:32px;font-size:15px;font-family:Inter,sans-serif'>"
    "Enter job posting details below to check if it looks real or fraudulent.</p>",
    unsafe_allow_html=True
)

# stats row
c1, c2, c3 = st.columns(3)
c1.metric("Training Samples", "17,880")
c2.metric("Real Jobs", "95.2%")
c3.metric("Feature Method", "TF-IDF")

st.markdown("<br>", unsafe_allow_html=True)

# ── form ──────────────────────────────────────────────────────────────────────
st.markdown("### 📋  Job Posting Details")

title = st.text_input("Job Title *", placeholder="e.g.  Senior Python Developer")

requirements = st.text_area(
    "Job Requirements / Description *",
    placeholder="Paste the full job requirements or description here...",
    height=160
)

col1, col2 = st.columns(2)
with col1:
    emp_type = st.selectbox(
        "Employment Type",
        ["Not Mentioned", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
    )
with col2:
    industry_toggle = st.selectbox("Industry", ["Not Mentioned", "Mentioned"])
    if industry_toggle == "Mentioned":
        industry = st.text_input("Specify Industry", placeholder="e.g.  Information Technology")
    else:
        industry = "Not Mentioned"

col3, col4 = st.columns(2)
with col3:
    location_toggle = st.selectbox("Location", ["Not Mentioned", "Mentioned"])
    if location_toggle == "Mentioned":
        location = st.text_input("Specify Location", placeholder="e.g.  US, CA, San Francisco")
    else:
        location = "Not Mentioned"
with col4:
    function_toggle = st.selectbox("Job Function", ["Not Mentioned", "Mentioned"])
    if function_toggle == "Mentioned":
        function_ = st.text_input("Specify Job Function", placeholder="e.g.  Engineering")
    else:
        function_ = "Not Mentioned"

st.markdown("<br>", unsafe_allow_html=True)
analyse = st.button("🔍  Analyse Job Posting")


# ── result ────────────────────────────────────────────────────────────────────
if analyse:
    if not title.strip() or not requirements.strip():
        st.error("⚠️  Please fill in at least the **Job Title** and **Requirements** fields.")
    else:
        with st.spinner("Analysing posting..."):
            verdict, confidence, rule_score, rule_hits = predict(title, requirements, emp_type, industry, function_)

        st.markdown("---")
        st.markdown("### 🔎 Result")

        req_clean  = clean_text(requirements)
        word_count = len(req_clean.split())
        char_count = len(req_clean)

        if verdict == REAL:
            st.markdown(
                f"""
                <div style='background:rgba(37,99,235,0.08);border:1px solid #3b82f6;
                            border-radius:4px;padding:28px 32px;margin-bottom:16px'>
                    <div style='font-size:2.2rem;margin-bottom:8px'>✅</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.8rem;
                                font-weight:800;color:#60a5fa;letter-spacing:-0.02em'>
                        Looks Real
                    </div>
                    <div style='font-family:monospace;font-size:11px;
                                color:#6b6b85;letter-spacing:0.1em;
                                text-transform:uppercase;margin-top:4px'>
                        Job posting appears legitimate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            bar_color    = "#00e5a0"
            verdict_label = "Real"

        elif verdict == SUSPICIOUS:
            st.markdown(
                f"""
                <div style='background:rgba(245,158,11,0.08);border:1px solid #f59e0b;
                            border-radius:4px;padding:28px 32px;margin-bottom:16px'>
                    <div style='font-size:2.2rem;margin-bottom:8px'>⚠️</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.8rem;
                                font-weight:800;color:#fbbf24;letter-spacing:-0.02em'>
                        Suspicious
                    </div>
                    <div style='font-family:monospace;font-size:11px;
                                color:#6b6b85;letter-spacing:0.1em;
                                text-transform:uppercase;margin-top:4px'>
                        Proceed with caution — verify before sharing any details
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            bar_color    = "#ffa500"
            verdict_label = "Suspicious"

        else:  # FRAUDULENT
            st.markdown(
                f"""
                <div style='background:rgba(239,68,68,0.08);border:1px solid #ef4444;
                            border-radius:4px;padding:28px 32px;margin-bottom:16px'>
                    <div style='font-size:2.2rem;margin-bottom:8px'>🚫</div>
                    <div style='font-family:Syne,sans-serif;font-size:1.8rem;
                                font-weight:800;color:#f87171;letter-spacing:-0.02em'>
                        Likely Fraudulent
                    </div>
                    <div style='font-family:monospace;font-size:11px;
                                color:#6b6b85;letter-spacing:0.1em;
                                text-transform:uppercase;margin-top:4px'>
                        Warning — do not apply or share personal details
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            bar_color    = "#ff4d6d"
            verdict_label = "Fraudulent"

        # risk score bar
        pct = min(rule_score / 8, 1.0)
        bar_clr = "#ef4444" if verdict == FRAUDULENT else ("#f59e0b" if verdict == SUSPICIOUS else "#3b82f6")
        st.markdown(
            f"<p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a6080;"
            f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px'>"
            f"Risk Score &nbsp;&middot;&nbsp; {rule_score} pts &nbsp;&nbsp;|&nbsp;&nbsp; Model Confidence &nbsp;&middot;&nbsp; {confidence:.1f}%</p>"
            f"<div style='background:#1a2742;border-radius:99px;height:7px;overflow:hidden;margin-bottom:6px'>"
            f"<div style='background:{bar_clr};width:{pct*100:.1f}%;height:100%;border-radius:99px'></div></div>",
            unsafe_allow_html=True
        )

        # score legend
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#2a3a55;letter-spacing:0.08em'>"
            "0–1 → Real &nbsp;&nbsp; 2–3 → Suspicious &nbsp;&nbsp; 4+ → Fraudulent</p>",
            unsafe_allow_html=True
        )

        # feature breakdown
        with st.expander("📊 Feature breakdown"):
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Word Count", word_count, help="Fake jobs tend to have shorter descriptions")
            fc2.metric("Char Count", char_count)
            fc3.metric("Risk Score", f"{rule_score} pts")

            st.markdown("<br>", unsafe_allow_html=True)
            industry_known = bool(industry.strip()) and industry.strip() != 'Not Mentioned'
            function_known = function_.strip() not in ('', 'Not Mentioned')
            st.markdown(
                f"| Feature | Value |\n"
                f"|---|---|\n"
                f"| Employment Type Known | {'✅ Yes' if emp_type != 'Not Mentioned' else '❌ No'} |\n"
                f"| Industry Provided     | {'✅ Yes' if industry_known else '❌ No'} |\n"
                f"| Function Provided     | {'✅ Yes' if function_known else '❌ No'} |\n"
                f"| Word Count            | {word_count} words |\n"
                f"| Char Count            | {char_count} characters |\n"
                f"| Rule Score            | {rule_score} pts |"
            )

        # flags section — show for Suspicious AND Fraudulent
        if verdict in (SUSPICIOUS, FRAUDULENT):
            st.markdown("---")
            if verdict == SUSPICIOUS:
                st.markdown("### ⚠️ Potential concerns detected")
                st.markdown(
                    "<p style='font-family:JetBrains Mono,monospace;font-size:12px;color:#fbbf24;line-height:1.6'>"
                    "These signals don't confirm fraud but warrant independent verification before proceeding."
                    "</p>", unsafe_allow_html=True
                )
            else:
                st.markdown("### 🚩 Red flags detected")

            flags = list(rule_hits)
            if word_count < 30:
                flags.append("Description is unusually brief — legitimate postings typically include detailed role expectations")
            if emp_type == 'Not Mentioned':
                flags.append("Employment type not specified — atypical for a formally listed position")
            if not industry.strip() or industry.strip() == 'Not Mentioned':
                flags.append("No industry specified — established organisations consistently identify their sector")
            if not flags:
                flags.append("ML model identified suspicious patterns in the text content")
            for flag in flags:
                icon = "⚠️" if verdict == SUSPICIOUS else "🔴"
                st.markdown(f"{icon} {flag}")

        st.markdown("---")
        st.markdown(
            """
            <div style='background:#0c1628;border:1px solid #1e2e4a;border-radius:10px;padding:18px 22px;margin-top:12px;border-left:3px solid #f59e0b'>
                <p style='font-family:JetBrains Mono,monospace;font-size:10px;color:#f59e0b;letter-spacing:0.12em;
                           text-transform:uppercase;margin-bottom:6px'>⚠️ Disclaimer</p>
                <p style='font-family:Inter,sans-serif;font-size:13px;color:#8a9cc0;line-height:1.7;margin:0'>
                    This tool is powered by a machine learning model with <strong style='color:#e8eeff'>93% accuracy</strong>
                    on test data — it is <strong style='color:#f87171'>not infallible</strong>. Always independently verify
                    job postings before sharing personal information or making any financial commitment.
                    Do not rely solely on this result. When in doubt, research the company directly,
                    check official websites, and consult trusted sources.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
