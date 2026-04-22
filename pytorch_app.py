"""
Surface Transport SLM — Government Accident Reporting Portal
=============================================================
Professional government-style interface for police officers,
RTO staff, and investigating officers.

Run: python -m streamlit run pytorch_app.py
"""

import streamlit as st
import os, sys, json, tempfile, re
import threading, queue, time

sys.path.insert(0, os.path.dirname(__file__))

from pytorch_ner import NERPredictor, auto_annotate, tokenize
from pytorch_qa import QAPredictor, FormGenerator, extract_from_text_regex
from pdf_processor import extract_text_from_pdf, clean_text

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Road Accident Reporting Tool — MoRTH, Government of India",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Professional Government Style ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── App Background ── */
.stApp { background: #f0f4f8; color: #1f2937; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1f3d !important;
}
/* Force ALL sidebar text to a readable light colour */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 12px 0 !important;
}
.sidebar-badge {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0 12px 0;
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 0.82rem;
    line-height: 1.7;
}
.sidebar-badge strong { color: #ffffff !important; }
.ipc-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 0.80rem;
}
.ipc-row .ipc-sec {
    font-weight: 700;
    color: #93c5fd !important;
    min-width: 40px;
    flex-shrink: 0;
}
.ipc-row .ipc-desc {
    color: #cbd5e1 !important;
}

/* ── Tricolor Strip ── */
.tricolor-strip {
    display: flex; height: 5px; width: 100%;
    margin-bottom: 0; overflow: hidden;
}
.tricolor-strip .saffron { flex: 1; background: #FF9933; }
.tricolor-strip .white   { flex: 1; background: #FFFFFF; border-top: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0; }
.tricolor-strip .green   { flex: 1; background: #138808; }

/* ── Top Banner ── */
.gov-banner {
    background: linear-gradient(135deg, #0b2244 0%, #1a3a6e 60%, #1f4b7a 100%);
    border-radius: 10px;
    padding: 20px 26px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(11,34,68,0.18);
    display: flex; justify-content: space-between; align-items: center;
}
.gov-banner h1 {
    font-size: 1.42rem; font-weight: 800;
    color: #ffffff !important; margin: 0; letter-spacing: -0.3px;
}
.gov-banner .sub-title {
    color: #a8c4e0 !important; margin: 4px 0 0 0;
    font-size: 0.80rem; font-weight: 400;
}
.gov-banner .act-ref {
    color: #a8c4e0 !important;
    font-size: 0.74rem; text-align: right; line-height: 1.6;
}
.gov-banner .status-badges {
    display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap;
}
.status-dot {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 3px 10px;
    color: #ffffff !important; font-size: 0.72rem; font-weight: 500;
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-green { background: #4ade80; box-shadow: 0 0 6px #4ade8088; }
.dot-amber { background: #fbbf24; box-shadow: 0 0 6px #fbbf2488; }

/* ── Step Cards ── */
.step-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px 16px;
    border: 1px solid #dde4ef;
    height: 100%;
}
.step-number {
    background: #0a1f3d;
    color: #ffffff;
    width: 26px; height: 26px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.76rem; font-weight: 700; margin-bottom: 10px;
}
.step-title { font-size: 0.88rem; font-weight: 700; color: #0a1f3d; margin-bottom: 5px; }
.step-desc  { font-size: 0.76rem; color: #5a6a7a; line-height: 1.55; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 8px !important;
    padding: 5px !important;
    border: 1px solid #dde4ef !important;
    gap: 3px !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5568 !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f0f4f8 !important;
    color: #0a1f3d !important;
}
.stTabs [aria-selected="true"] {
    background: #0a1f3d !important;
    color: #ffffff !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div { color: #ffffff !important; }
.stTabs [aria-selected="false"] p,
.stTabs [aria-selected="false"] span,
.stTabs [aria-selected="false"] div { color: #4a5568 !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Content Cards ── */
.content-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px 22px;
    border: 1px solid #dde4ef;
    margin-bottom: 14px;
}

/* ── Section Heading ── */
.sec-head {
    font-size: 0.96rem; font-weight: 700; color: #0a1f3d;
    margin: 0 0 14px 0;
    padding-bottom: 9px;
    border-bottom: 2px solid #FF9933;
}

/* ── Entity Chips ── */
.chip {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 12px; border-radius: 4px; margin: 3px;
    font-size: 0.77rem; font-weight: 600;
}
.chip-label { font-weight: 500; color: rgba(0,0,0,0.40); font-size: 0.71rem; }
.chip-red    { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.chip-orange { background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa; }
.chip-green  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.chip-blue   { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
.chip-purple { background: #faf5ff; color: #6b21a8; border: 1px solid #ddd6fe; }
.chip-gray   { background: #f8fafc; color: #374151; border: 1px solid #e2e8f0; }
.chip-teal   { background: #f0fdfa; color: #134e4a; border: 1px solid #99f6e4; }

/* ── Stat Boxes ── */
.stat-box {
    border-radius: 10px; padding: 14px 12px;
    text-align: center; overflow: hidden;
}
.stat-box .num {
    font-size: 1.4rem; font-weight: 800; line-height: 1.2;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.stat-box .lbl {
    font-size: 0.62rem; font-weight: 700; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 0.6px;
}
.stat-red    { background: #fef2f2; border: 1px solid #fecaca; }
.stat-red    .num { color: #dc2626; }
.stat-red    .lbl { color: #991b1b; }
.stat-orange { background: #fff7ed; border: 1px solid #fed7aa; }
.stat-orange .num { color: #ea580c; }
.stat-orange .lbl { color: #9a3412; }
.stat-blue   { background: #eff6ff; border: 1px solid #bfdbfe; }
.stat-blue   .num { color: #1d4ed8; }
.stat-blue   .lbl { color: #1e40af; }
.stat-green  { background: #f0fdf4; border: 1px solid #bbf7d0; }
.stat-green  .num { color: #16a34a; }
.stat-green  .lbl { color: #166534; }

/* ── Answer Card ── */
.answer-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    border: 1px solid #dde4ef;
    border-left: 4px solid #0a1f3d;
}
.answer-card .ans-num {
    font-size: 0.66rem; font-weight: 700; color: #0a1f3d;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 7px;
    display: flex; align-items: center; justify-content: space-between;
}
.answer-card .ans-text { font-size: 0.86rem; color: #2d3748; line-height: 1.72; }
.answer-card .ans-src  {
    font-size: 0.71rem; color: #94a3b8; margin-top: 10px;
    border-top: 1px solid #f1f5f9; padding-top: 7px;
}
.score-bar {
    height: 4px; border-radius: 2px; background: #e2e8f0;
    margin-left: 6px; overflow: hidden; width: 70px; display: inline-block;
    vertical-align: middle;
}
.score-fill { height: 100%; background: #0a1f3d; border-radius: 2px; }

/* ── Alert Boxes ── */
.info-box {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6; border-radius: 6px;
    padding: 12px 16px; font-size: 0.82rem; color: #1e40af; line-height: 1.55;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b; border-radius: 6px;
    padding: 12px 16px; font-size: 0.82rem; color: #92400e; line-height: 1.55;
}
.success-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #22c55e; border-radius: 6px;
    padding: 12px 16px; font-size: 0.82rem; color: #166534; line-height: 1.55;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: #0a1f3d !important;
    border: none !important;
    color: #ffffff !important;
    padding: 10px 22px !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #1a3a6e !important;
    color: #ffffff !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[data-testid="stBaseButton-primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] span {
    color: #ffffff !important;
}
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: #ffffff !important;
    border: 1px solid #d0d5dd !important;
    color: #2d3748 !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    border-radius: 8px !important;
    border: 1px solid #d4dce8 !important;
    font-size: 0.86rem !important;
    background: #ffffff !important;
    color: #1a1a2e !important;
    padding: 9px 12px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #3b6fa0 !important;
    box-shadow: 0 0 0 2px rgba(59,111,160,0.12) !important;
}
.stApp label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    color: #166534 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: #dcfce7 !important; }
.stDownloadButton > button p,
.stDownloadButton > button span { color: #166534 !important; }

/* ── Radio ── */
.stRadio > div { gap: 10px !important; }
.stRadio label { font-weight: 500 !important; font-size: 0.83rem !important; }

/* ── Selectbox ── */
[data-baseweb="select"] { border-radius: 6px !important; }

/* ── Divider ── */
hr { border-color: #dce3ec !important; }

/* ── Global main content text ── */
.stApp .stMarkdown p,
.stApp .stMarkdown span,
.stApp .stMarkdown li,
.stApp .stRadio label,
.stApp .stRadio div,
.stApp .stTextArea label,
.stApp .stTextInput label,
.stApp .stExpander summary span {
    color: #2d3748 !important;
}
/* Banner text overrides */
.stApp .gov-banner h1 { color: #ffffff !important; }
.stApp .gov-banner .sub-title { color: #a8c4e0 !important; }
.stApp .gov-banner p,
.stApp .gov-banner span,
.stApp .status-dot span { color: #ffffff !important; }

/* ── Empty State ── */
.empty-state {
    background: #f8fafc;
    border: 2px dashed #c8d6e8;
    border-radius: 10px; padding: 56px 32px; text-align: center;
}
.empty-state .e-title {
    font-size: 0.96rem; font-weight: 700; color: #475569; margin-bottom: 6px;
}
.empty-state .e-sub { font-size: 0.80rem; color: #94a3b8; line-height: 1.55; }

/* ── Form Section Titles ── */
.form-section-title {
    font-size: 0.82rem; font-weight: 700; color: #0a1f3d;
    padding: 9px 14px;
    background: #eef2f7;
    border-left: 3px solid #FF9933;
    border-radius: 0 6px 6px 0;
    margin: 14px 0 10px 0;
}

/* ── Model Status Card ── */
.model-status-card {
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
    display: flex; align-items: flex-start; gap: 12px;
}
.model-status-ready  { background: #f0fdf4; border: 1px solid #86efac; }
.model-status-notready { background: #fffbeb; border: 1px solid #fde68a; }
.ms-title { font-size: 0.88rem; font-weight: 700; }
.ms-sub   { font-size: 0.74rem; margin-top: 2px; }
.ms-ready .ms-title { color: #166534; }
.ms-ready .ms-sub   { color: #4ade80; }
.ms-warn  .ms-title { color: #92400e; }
.ms-warn  .ms-sub   { color: #d97706; }

/* ── Footer ── */
.gov-footer {
    background: #0a1f3d;
    color: #8ea9cc !important;
    padding: 14px 28px;
    font-size: 0.71rem;
    text-align: center;
    margin-top: 36px;
    border-radius: 10px;
}
.gov-footer * { color: #8ea9cc !important; }

/* ── Misc ── */
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
.stSpinner > div { color: #0a1f3d !important; }
.stExpander { border-radius: 8px !important; border: 1px solid #dde4ef !important; }
[data-testid="stNumberInput"] input { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_ner():
    p = NERPredictor(model_dir="models")
    if os.path.exists("models/ner_bilstm_crf.pt"):
        p.load()
        return p, True
    return p, False

@st.cache_resource
def load_qa():
    p = QAPredictor(model_dir="models")
    if os.path.exists("models/qa_neural.pt"):
        p.load()
        return p, True
    return p, False

ner_pred, ner_ready = load_ner()
qa_pred,  qa_ready  = load_qa()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Quick Reference")
    st.markdown("---")

    st.markdown("**FAR — Form 1**")
    st.markdown("""
<div class="sidebar-badge">
First Accident Report<br>
<strong>Deadline:</strong> 48 hours<br>
<strong>Filed by:</strong> Investigating Officer<br>
<strong>Submitted to:</strong> MACT
</div>""", unsafe_allow_html=True)

    st.markdown("**DAR — Form VII**")
    st.markdown("""
<div class="sidebar-badge">
Detailed Accident Report<br>
<strong>Deadline:</strong> 90 days<br>
<strong>Filed by:</strong> Investigating Officer<br>
<strong>Submitted to:</strong> Claims Tribunal
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Key IPC Sections**")
    ipc_items = [
        ("279",  "Rash driving"),
        ("304A", "Causing death by negligence"),
        ("337",  "Hurt by rash act"),
        ("338",  "Grievous hurt by rash act"),
        ("304B", "Dowry death provision"),
    ]
    ipc_html = "".join(
        f'<div class="ipc-row"><span class="ipc-sec">{sec}</span>'
        f'<span class="ipc-desc">{desc}</span></div>'
        for sec, desc in ipc_items
    )
    st.markdown(ipc_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Status**")
    ner_icon = "Ready" if ner_ready else "Not trained"
    qa_icon  = "Ready" if qa_ready  else "Not trained"
    st.markdown(f"NER Model: **{ner_icon}**")
    st.markdown(f"QA Model: **{qa_icon}**")

    st.markdown("---")
    st.markdown("**Usage Tips**")
    st.markdown("""
- Paste text from news articles, FIRs, or press releases
- PDF uploads extract text automatically
- Review and edit all auto-filled fields before downloading
- Use the Q&A tab to look up regulations
""")

# ── Top Banner ────────────────────────────────────────────────
ner_dot   = "dot-green" if ner_ready else "dot-amber"
qa_dot    = "dot-green" if qa_ready  else "dot-amber"
ner_label = "NER Model: Ready" if ner_ready else "NER Model: Not Trained"
qa_label  = "QA Model: Ready"  if qa_ready  else "QA Model: Not Trained"

st.markdown("""
<div class="tricolor-strip">
    <div class="saffron"></div><div class="white"></div><div class="green"></div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="gov-banner">
    <div>
        <h1>Road Accident Reporting Tool</h1>
        <p class="sub-title">Ministry of Road Transport &amp; Highways — Government of India</p>
        <div class="status-badges">
            <span class="status-dot"><span class="dot {ner_dot}"></span> {ner_label}</span>
            <span class="status-dot"><span class="dot {qa_dot}"></span> {qa_label}</span>
            <span class="status-dot"><span class="dot dot-green"></span> System Online</span>
        </div>
    </div>
    <div class="act-ref">Motor Vehicles Act, 1988<br>Section 158(6)</div>
</div>
""", unsafe_allow_html=True)

# ── Model warning ─────────────────────────────────────────────
if not ner_ready:
    st.markdown("""
    <div class="warn-box">
        <strong>Note:</strong> AI model not trained. Pattern-based extraction is active.
        Go to <strong>Settings &amp; Training</strong> to train the AI model for higher accuracy.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, title, desc in [
    (c1, "1", "Paste or Upload Text",  "Paste any accident report, FIR text, or news article — or upload a PDF document"),
    (c2, "2", "AI Extraction",         "The BiLSTM-CRF model extracts entities: location, vehicles, casualties, IPC sections"),
    (c3, "3", "Review & Edit Form",    "Extracted fields appear in a pre-filled FAR / DAR form for you to verify and correct"),
    (c4, "4", "Download Report",       "Download as PDF, plain text, or JSON — ready for submission to the Claims Tribunal"),
]:
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Fill Accident Report  ",
    "  Ask a Question  ",
    "  Blank Forms  ",
    "  Settings & Training  ",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — FILL ACCIDENT REPORT
# ═══════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="sec-head">Paste Accident Text</div>', unsafe_allow_html=True)

        mode = st.radio("Input method:",
                        ["Paste or type text", "Upload a PDF file"],
                        label_visibility="collapsed")

        if "Paste" in mode:
            sample = (
                "FIR number RJ/PS/045/2026 was registered on 22 April 2026 at "
                "Mathania Police Station, Jodhpur under Sections 279 and 304A of IPC. "
                "The accident occurred on Friday, 18 April 2026 at 3:30 PM on "
                "State Highway 61 near Mathania Bus Stand in Jodhpur, Rajasthan."
            )
            text = st.text_area(
                "Paste the accident news or report text here:",
                value=sample, height=260,
                help="Paste from any news article, FIR, or press release"
            )
        else:
            pdf_up = st.file_uploader(
                "Upload a PDF accident report or news article:",
                type=["pdf"], key="ner_pdf"
            )
            text = ""
            if pdf_up:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(pdf_up.read())
                with st.spinner("Reading PDF..."):
                    res  = extract_text_from_pdf(tmp.name)
                    text = clean_text(res.get("full_text", ""))
                os.unlink(tmp.name)
                st.markdown(
                    f'<div class="success-box">PDF read successfully — '
                    f'<strong>{len(text):,}</strong> characters extracted.</div>',
                    unsafe_allow_html=True
                )
                st.text_area("Extracted text (preview):", value=text[:1500],
                             height=200, disabled=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Analyse Text & Fill Report",
                            type="primary", use_container_width=True)

        with st.expander("Sample accident texts"):
            samples = {
                "Highway pile-up": (
                    "Eight persons died and twenty three were injured in a pile-up "
                    "involving three trucks and two cars on NH48 near Vadodara in "
                    "Gujarat on Sunday. Overspeeding and brake failure were the cause."
                ),
                "Hit and run": (
                    "A hit-and-run accident on State Highway 17 near Kolhapur in "
                    "Maharashtra killed two pedestrians and injured four others on "
                    "Wednesday night. Poor visibility due to heavy fog was cited. "
                    "Police registered a case under Section 279 and 338 of the IPC."
                ),
                "Bus accident": (
                    "Fourteen passengers were killed and thirty two injured when a "
                    "private bus overturned on NH66 near Ratnagiri in Maharashtra "
                    "on Thursday due to rash driving near a sharp curve in heavy rain."
                ),
            }
            for name, s_text in samples.items():
                if st.button(f"Use: {name}", key=f"sample_{name}",
                             use_container_width=True):
                    st.session_state["sample_text"] = s_text
                    st.rerun()

        if "sample_text" in st.session_state:
            text = st.session_state.pop("sample_text")
            run_btn = True

    with right:
        if run_btn and text:
            with st.spinner("Extracting details from text..."):
                if ner_ready:
                    entities     = ner_pred.extract_entities(text)
                    far_data_ner = ner_pred.extract_to_far_format(text)
                else:
                    annotated = auto_annotate(text)
                    entities  = {}
                    for tok, tag in annotated:
                        if tag != "O":
                            etype = tag.split("-", 1)[1]
                            entities.setdefault(etype, [])
                            if tag.startswith("B-"):   entities[etype].append(tok)
                            elif entities[etype]:       entities[etype][-1] += " " + tok
                    far_data_ner = {}

                regex_data = extract_from_text_regex(text)

            if ner_ready:
                st.markdown("""
                <div class="success-box">
                    <strong>Extraction complete (AI model).</strong>
                    Fields populated automatically using the BiLSTM-CRF model.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>Extraction complete (pattern-based).</strong>
                    Train the AI model in the Settings tab for higher accuracy.
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Quick stats
            def _pick(regex_key, ner_key, default="—"):
                v = regex_data.get(regex_key, "")
                if v and v != "0": return v
                v2 = far_data_ner.get(ner_key, "")
                if v2 and v2 != "0": return v2
                if regex_data.get(regex_key) == "0": return "0"
                return default

            killed  = _pick("num_fatalities",   "number_of_fatalities", "0")
            injured = _pick("num_injured",       "number_of_injured",    "0")
            loc     = _pick("place_of_accident", "place_of_accident",    "—")
            nature  = _pick("nature_of_accident","nature_of_accident",   "—")

            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, val, lbl, cls in [
                (sc1, killed,  "Killed",   "stat-red"),
                (sc2, injured, "Injured",  "stat-orange"),
                (sc3, str(loc)[:14] + ("..." if len(str(loc)) > 14 else ""), "Location", "stat-blue"),
                (sc4, nature,  "Nature",   "stat-green"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="stat-box {cls}">
                        <div class="num" title="{val}">{val}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Entity chips
            if entities:
                st.markdown('<div class="sec-head">Extracted Entities</div>',
                            unsafe_allow_html=True)
                FRIENDLY = {
                    "NUM_KILLED":   ("Killed",      "chip-red"),
                    "NUM_INJURED":  ("Injured",     "chip-orange"),
                    "LOCATION":     ("Place",       "chip-blue"),
                    "STATE":        ("State",       "chip-green"),
                    "VEHICLE":      ("Vehicle",     "chip-blue"),
                    "COLLISION":    ("Collision",   "chip-purple"),
                    "CAUSE":        ("Cause",       "chip-orange"),
                    "ROAD_TYPE":    ("Road",        "chip-gray"),
                    "DATE":         ("Date",        "chip-teal"),
                    "TIME":         ("Time",        "chip-teal"),
                    "WEATHER":      ("Weather",     "chip-blue"),
                    "IPC_SECTION":  ("IPC Section", "chip-gray"),
                    "REG_NUMBER":   ("Reg. No.",    "chip-green"),
                    "VICTIM_TYPE":  ("Victim",      "chip-gray"),
                    "SEVERITY":     ("Severity",    "chip-red"),
                }
                ORDER = ["NUM_KILLED","NUM_INJURED","LOCATION","STATE","VEHICLE",
                         "COLLISION","CAUSE","ROAD_TYPE","DATE","TIME",
                         "IPC_SECTION","REG_NUMBER","WEATHER","SEVERITY","VICTIM_TYPE"]
                chips_html = ""
                for etype in ORDER:
                    if etype not in entities: continue
                    label, css = FRIENDLY.get(etype, (etype, "chip-gray"))
                    for v in entities[etype]:
                        chips_html += (
                            f'<span class="chip {css}">'
                            f'<span class="chip-label">{label}:</span>'
                            f'<strong>{v}</strong></span>'
                        )
                st.markdown(f'<div style="line-height:2.6;">{chips_html}</div>',
                            unsafe_allow_html=True)

            # Editable form
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-head">Edit & Download Report</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                Review and correct the fields below before downloading.
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            merged = dict(regex_data)
            if far_data_ner:
                ner_map = {
                    'date_of_accident':   'date_of_accident',
                    'time_of_accident':   'time_of_accident',
                    'place_of_accident':  'place_of_accident',
                    'state':              'state',
                    'number_of_fatalities': 'num_fatalities',
                    'number_of_injured':  'num_injured',
                    'vehicle_1':          'v1_type',
                    'vehicle_2':          'v2_type',
                    'road_type':          'road_type',
                    'weather_condition':  'weather',
                    'collision_type':     'collision_type',
                    'cause_of_accident':  'cause',
                    'ipc_sections':       'ipc_sections',
                    'registration_number':'v1_reg',
                }
                for ner_key, merge_key in ner_map.items():
                    if ner_key in far_data_ner and far_data_ner[ner_key] and not merged.get(merge_key):
                        merged[merge_key] = far_data_ner[ner_key]

            report_type = st.radio(
                "Report Type",
                ["FAR (First Accident Report)", "DAR (Detailed Accident Report)"],
                horizontal=True, key="report_type_tab1"
            )

            st.markdown('<div class="form-section-title">A. Case Information</div>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            with fc1:
                merged['fir_no']           = st.text_input("FIR No.",            value=merged.get('fir_no',''),           key="t1_fir")
                merged['date_of_accident'] = st.text_input("Date of Accident",   value=merged.get('date_of_accident',''), key="t1_doa")
                merged['police_station']   = st.text_input("Police Station",     value=merged.get('police_station',''),   key="t1_ps")
            with fc2:
                merged['report_date']      = st.text_input("Date of Report",     value=merged.get('report_date',''),      key="t1_dor")
                merged['time_of_accident'] = st.text_input("Time of Accident",   value=merged.get('time_of_accident',''), key="t1_toa")
                merged['ipc_sections']     = st.text_input("IPC / BNS Sections", value=merged.get('ipc_sections',''),     key="t1_ipc")

            st.markdown('<div class="form-section-title">B. Accident Details</div>', unsafe_allow_html=True)
            fc3, fc4 = st.columns(2)
            with fc3:
                merged['place_of_accident']  = st.text_input("Place of Accident", value=merged.get('place_of_accident',''), key="t1_place")
                merged['state']              = st.text_input("State",              value=merged.get('state',''),             key="t1_state")
                nat_opts = ["Fatal", "Grievous Injury", "Injury", "Damage/Loss of property"]
                merged['nature_of_accident'] = st.selectbox(
                    "Nature of Accident", nat_opts,
                    index=nat_opts.index(merged.get('nature_of_accident','Fatal'))
                          if merged.get('nature_of_accident','Fatal') in nat_opts else 0,
                    key="t1_nature"
                )
            with fc4:
                merged['num_fatalities'] = st.text_input("Persons Killed",    value=merged.get('num_fatalities','0'), key="t1_killed")
                merged['num_injured']    = st.text_input("Persons Injured",   value=merged.get('num_injured','0'),    key="t1_injured")
                merged['num_vehicles']   = st.text_input("Vehicles Involved", value=merged.get('num_vehicles',''),    key="t1_veh")

            st.markdown('<div class="form-section-title">C. Vehicle Details</div>', unsafe_allow_html=True)
            fc5, fc6 = st.columns(2)
            with fc5:
                st.markdown("**Vehicle 1**")
                merged['v1_type']      = st.text_input("Type",              value=merged.get('v1_type',''),      key="t1_v1type")
                merged['v1_reg']       = st.text_input("Registration No.",  value=merged.get('v1_reg',''),       key="t1_v1reg")
                merged['v1_driver']    = st.text_input("Driver Name",       value=merged.get('v1_driver',''),    key="t1_v1drv")
                merged['v1_owner']     = st.text_input("Owner",             value=merged.get('v1_owner',''),     key="t1_v1own")
                merged['v1_insurance'] = st.text_input("Insurance Co.",     value=merged.get('v1_insurance',''), key="t1_v1ins")
            with fc6:
                st.markdown("**Vehicle 2**")
                merged['v2_type']      = st.text_input("Type",              value=merged.get('v2_type',''),      key="t1_v2type")
                merged['v2_reg']       = st.text_input("Registration No.",  value=merged.get('v2_reg',''),       key="t1_v2reg")
                merged['v2_driver']    = st.text_input("Driver Name",       value=merged.get('v2_driver',''),    key="t1_v2drv")
                merged['v2_owner']     = st.text_input("Owner",             value=merged.get('v2_owner',''),     key="t1_v2own")
                merged['v2_insurance'] = st.text_input("Insurance Co.",     value=merged.get('v2_insurance',''), key="t1_v2ins")

            st.markdown('<div class="form-section-title">D. Road & Conditions</div>', unsafe_allow_html=True)
            fc7, fc8 = st.columns(2)
            with fc7:
                merged['collision_type'] = st.text_input("Collision Type",     value=merged.get('collision_type',''), key="t1_ctype")
                merged['cause']          = st.text_input("Cause of Accident",  value=merged.get('cause',''),          key="t1_cause")
                merged['road_type']      = st.text_input("Road Type",          value=merged.get('road_type',''),      key="t1_road")
            with fc8:
                merged['weather']    = st.text_input("Weather",    value=merged.get('weather',''),    key="t1_weather")
                merged['lighting']   = st.text_input("Lighting",   value=merged.get('lighting',''),   key="t1_light")
                merged['visibility'] = st.text_input("Visibility", value=merged.get('visibility',''), key="t1_vis")

            st.markdown('<div class="form-section-title">E. Investigation</div>', unsafe_allow_html=True)
            fc9, fc10 = st.columns(2)
            with fc9:
                merged['officer_name'] = st.text_input("Investigating Officer", value=merged.get('officer_name',''), key="t1_io")
                merged['officer_pis']  = st.text_input("PIS Number",            value=merged.get('officer_pis',''),  key="t1_pis")
            with fc10:
                merged['hospital'] = st.text_input("Hospital", value=merged.get('hospital',''), key="t1_hosp")
                merged['doctor']   = st.text_input("Doctor",   value=merged.get('doctor',''),   key="t1_doc")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="form-section-title">Download Report</div>', unsafe_allow_html=True)
            dl1, dl2, dl3 = st.columns(3)

            try:
                from pdf_generator import generate_far_pdf, generate_dar_pdf
                pdf_available = True
            except ImportError:
                pdf_available = False

            with dl1:
                if pdf_available:
                    if "FAR" in report_type:
                        pdf_bytes = generate_far_pdf(merged)
                        st.download_button("Download FAR as PDF",
                                           data=pdf_bytes, file_name="FAR_Report.pdf",
                                           mime="application/pdf",
                                           use_container_width=True,
                                           key="tab1_download_far_pdf")
                    else:
                        pdf_bytes = generate_dar_pdf(merged)
                        st.download_button("Download DAR as PDF",
                                           data=pdf_bytes, file_name="DAR_Report.pdf",
                                           mime="application/pdf",
                                           use_container_width=True,
                                           key="tab1_download_dar_pdf")
            with dl2:
                fg   = FormGenerator()
                form = fg.get_far_template() if "FAR" in report_type else fg.get_dar_template()
                for field in form["fields"]:
                    if field["id"] in merged and merged[field["id"]]:
                        field["value"] = str(merged[field["id"]])
                form_text = fg.form_to_text(form)
                st.download_button("Download as Text",
                                   data=form_text,
                                   file_name=f"{'FAR' if 'FAR' in report_type else 'DAR'}_Report.txt",
                                   use_container_width=True,
                                   key="tab1_download_report_text")
            with dl3:
                st.download_button("Download as JSON",
                                   data=json.dumps(merged, indent=2, ensure_ascii=False),
                                   file_name=f"{'FAR' if 'FAR' in report_type else 'DAR'}_data.json",
                                   use_container_width=True,
                                   key="tab1_download_report_json")

        elif run_btn and not text:
            st.markdown("""
            <div class="warn-box">Please paste some text or upload a PDF first.</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="e-title">Your extracted report will appear here</div>
                <div class="e-sub">Paste accident text on the left and click
                    <strong>Analyse Text &amp; Fill Report</strong> to begin.</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — ASK A QUESTION
# ═══════════════════════════════════════════════════════════════
with tab2:
    ql, qr = st.columns([1, 1], gap="large")

    with ql:
        st.markdown('<div class="sec-head">Ask About Traffic Laws & Regulations</div>',
                    unsafe_allow_html=True)

        if not qa_ready:
            st.markdown("""
            <div class="warn-box">
                <strong>QA model not trained.</strong> Go to the
                <strong>Settings &amp; Training</strong> tab to upload documents and train.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                Ask any question about the Motor Vehicles Act, accident reporting procedures,
                road safety guidelines, or transport regulations.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            q = st.text_input("Type your question:",
                              placeholder="e.g. What is the time limit to file a First Accident Report?",
                              label_visibility="collapsed")

            st.markdown("**Common questions** — click to pre-fill:")
            sample_qs = [
                "What is a fatal accident?",
                "What is the time limit for filing a FAR report?",
                "Who is responsible for filing the accident report?",
                "What information is required in the accident report?",
                "What is the Motor Vehicles Act?",
                "What are the rules for hit and run accidents?",
            ]
            q_cols = st.columns(2)
            for i, sq in enumerate(sample_qs):
                with q_cols[i % 2]:
                    if st.button(sq, key=f"q_{i}", use_container_width=True):
                        q = sq

            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("Search Regulations", type="primary",
                                   use_container_width=True,
                                   disabled=not bool(q and q.strip()))

    with qr:
        if not qa_ready:
            st.markdown("""
            <div class="empty-state">
                <div class="e-title">Regulatory Q&A</div>
                <div class="e-sub">Train the QA model first in the Settings tab.</div>
            </div>""", unsafe_allow_html=True)
        elif q:
            with st.spinner("Searching transport law database..."):
                results = qa_pred.answer(q, top_k=3)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'<div class="sec-head">Results for: '
                f'<em style="font-weight:400">{q}</em></div>',
                unsafe_allow_html=True
            )

            if results:
                for i, r in enumerate(results):
                    score     = r.get('score', 0)
                    score_pct = min(int(score * 100), 100) if score <= 1 else min(int(score * 10), 100)
                    score_bar = (
                        f'<div class="score-bar">'
                        f'<div class="score-fill" style="width:{score_pct}%"></div>'
                        f'</div>'
                    )
                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="ans-num">
                            <span>Answer {i+1} of {len(results)}</span>
                            <span>Score: {score:.2f} {score_bar}</span>
                        </div>
                        <div class="ans-text">{r['answer']}</div>
                        <div class="ans-src">Source: <i>{r.get('source','Knowledge Base')}</i></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="font-size:0.77rem;color:#94a3b8;margin-top:10px;
                            background:#f8fafc;border-radius:6px;padding:9px 14px;
                            border:1px solid #e2e8f0;">
                    Answers are extracted from uploaded transport law documents.
                    Verify with official sources before taking any legal action.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    No highly relevant answers found. Try rephrasing your question.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="e-title">Regulatory Q&A</div>
                <div class="e-sub">Type a question on the left or choose from
                    the common questions to search the law database.</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — BLANK FORMS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">Download Blank / Pre-filled Forms</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Generate a blank template or auto-fill from accident text.
        Review each field, then download as PDF or text for submission.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:0.94rem;font-weight:700;color:#003366;margin-bottom:6px;">
                First Accident Report (FAR) — Form 1
            </div>
            <div style="font-size:0.81rem;color:#555;">
                Must be filed within <strong>48 hours</strong> of the accident.
                Submitted to the Motor Accidents Claims Tribunal (MACT).
            </div>
        </div>
        """, unsafe_allow_html=True)

        autofill_1 = st.text_area("Auto-fill from accident text (optional):",
                                  height=100, key="far_fill",
                                  placeholder="Paste accident text here to auto-fill fields...")

        if st.button("Generate FAR Form", type="primary",
                     use_container_width=True, key="gen_far"):
            fg = FormGenerator()
            if autofill_1.strip():
                form, regex_d = fg.prefill_far_from_regex(autofill_1)
                if ner_ready:
                    ner_local, ner_ok2 = load_ner()
                    if ner_ok2:
                        ext      = ner_local.extract_to_far_format(autofill_1)
                        form_ner = fg.prefill_far(ext)
                        for i, field in enumerate(form["fields"]):
                            if not field.get("value") and form_ner["fields"][i].get("value"):
                                field["value"] = form_ner["fields"][i]["value"]
            else:
                form    = fg.get_far_template()
                regex_d = {}

            st.markdown(
                f'<div class="form-section-title">{form["form_title"]}</div>',
                unsafe_allow_html=True
            )
            vals = {}
            for field in form["fields"]:
                if field["type"] == "select":
                    opts = field.get("options", [])
                    idx  = opts.index(field["value"]) if field["value"] in opts else 0
                    vals[field["id"]] = st.selectbox(field["label"], opts, index=idx,
                                                     key=f"far_{field['id']}")
                elif field["type"] == "number":
                    try:    v = int(field.get("value","") or 0)
                    except: v = 0
                    vals[field["id"]] = st.number_input(field["label"], value=v,
                                                        key=f"far_{field['id']}")
                elif field["type"] == "textarea":
                    vals[field["id"]] = st.text_area(field["label"],
                                                     value=field.get("value",""),
                                                     key=f"far_{field['id']}")
                else:
                    vals[field["id"]] = st.text_input(field["label"],
                                                      value=field.get("value",""),
                                                      key=f"far_{field['id']}")

            pdf_data = dict(regex_d)
            pdf_data.update({k: str(v) for k, v in vals.items() if v})
            lines = [form["form_title"], "="*50]
            for field in form["fields"]:
                lines.append(f"{field['label']}: {vals.get(field['id'],'')}")

            dc1, dc2 = st.columns(2)
            with dc1:
                try:
                    from pdf_generator import generate_far_pdf
                    far_pdf = generate_far_pdf(pdf_data)
                    st.download_button("Download FAR as PDF",
                                       data=far_pdf, file_name="FAR_Report.pdf",
                                       mime="application/pdf",
                                       use_container_width=True,
                                       key="tab3_far_download_pdf")
                except Exception as e:
                    st.error(f"PDF unavailable: {e}")
            with dc2:
                st.download_button("Download FAR (.txt)",
                                   data="\n".join(lines), file_name="FAR_filled.txt",
                                   use_container_width=True,
                                   key="tab3_far_download_txt")

    with f2:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:0.94rem;font-weight:700;color:#003366;margin-bottom:6px;">
                Detailed Accident Report (DAR) — Form VII
            </div>
            <div style="font-size:0.81rem;color:#555;">
                Must be filed within <strong>90 days</strong> of the accident.
                Contains complete investigation details including driver and victim information.
            </div>
        </div>
        """, unsafe_allow_html=True)

        autofill_2 = st.text_area("Auto-fill from accident text (optional):",
                                  height=100, key="dar_fill",
                                  placeholder="Paste accident text here to auto-fill fields...")

        if st.button("Generate DAR Form", type="primary",
                     use_container_width=True, key="gen_dar"):
            fg = FormGenerator()
            if autofill_2.strip():
                form, regex_d = fg.prefill_dar_from_regex(autofill_2)
            else:
                form    = fg.get_dar_template()
                regex_d = {}

            st.markdown(
                f'<div class="form-section-title">{form["form_title"]}</div>',
                unsafe_allow_html=True
            )
            vals = {}
            for field in form["fields"]:
                if field["type"] == "select":
                    opts = field.get("options", [])
                    idx  = opts.index(field["value"]) if field["value"] in opts else 0
                    vals[field["id"]] = st.selectbox(field["label"], opts, index=idx,
                                                     key=f"dar_{field['id']}")
                elif field["type"] == "number":
                    try:    v = int(field.get("value","") or 0)
                    except: v = 0
                    vals[field["id"]] = st.number_input(field["label"], value=v,
                                                        key=f"dar_{field['id']}")
                elif field["type"] == "textarea":
                    vals[field["id"]] = st.text_area(field["label"],
                                                     value=field.get("value",""),
                                                     key=f"dar_{field['id']}")
                else:
                    vals[field["id"]] = st.text_input(field["label"],
                                                      value=field.get("value",""),
                                                      key=f"dar_{field['id']}")

            pdf_data = dict(regex_d)
            pdf_data.update({k: str(v) for k, v in vals.items() if v})
            lines = [form["form_title"], "="*50]
            for field in form["fields"]:
                lines.append(f"{field['label']}: {vals.get(field['id'],'')}")

            dc3, dc4 = st.columns(2)
            with dc3:
                try:
                    from pdf_generator import generate_dar_pdf
                    dar_pdf = generate_dar_pdf(pdf_data)
                    st.download_button("Download DAR as PDF",
                                       data=dar_pdf, file_name="DAR_Report.pdf",
                                       mime="application/pdf",
                                       use_container_width=True,
                                       key="tab3_dar_download_pdf")
                except Exception as e:
                    st.error(f"PDF unavailable: {e}")
            with dc4:
                st.download_button("Download DAR (.txt)",
                                   data="\n".join(lines), file_name="DAR_filled.txt",
                                   use_container_width=True,
                                   key="tab3_dar_download_txt")

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS & TRAINING
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">System Status</div>', unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        ms_cls  = "model-status-ready ms-ready" if ner_ready else "model-status-notready ms-warn"
        ms_lbl  = "NER Model (Accident Analysis) — Ready" if ner_ready else "NER Model — Not Trained"
        ms_sub  = "BiLSTM-CRF entity extraction active" if ner_ready else "Upload PDFs and train to enable"
        st.markdown(f"""
        <div class="model-status-card {ms_cls}">
            <div class="ms-info">
                <div class="ms-title">{ms_lbl}</div>
                <div class="ms-sub">{ms_sub}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if ner_ready:
            try:
                import torch
                ck = torch.load("models/ner_bilstm_crf.pt", map_location="cpu", weights_only=False)
                st.caption(f"{ck['total_params']:,} parameters — BiLSTM-CRF architecture")
            except: pass

    with s2:
        ms_cls  = "model-status-ready ms-ready" if qa_ready else "model-status-notready ms-warn"
        ms_lbl  = "QA Model (Regulations) — Ready" if qa_ready else "QA Model — Not Trained"
        ms_sub  = "Dual encoder neural retrieval active" if qa_ready else "Upload PDFs and train to enable"
        st.markdown(f"""
        <div class="model-status-card {ms_cls}">
            <div class="ms-info">
                <div class="ms-title">{ms_lbl}</div>
                <div class="ms-sub">{ms_sub}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("Model Architecture"):
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("""
**NER Model — BiLSTM-CRF**
- Embedding: Vocab × 128 dims
- BiLSTM: 2 layers, 256 hidden (bidirectional)
- CRF layer with Viterbi decoding
- BIO tagging scheme, ~538K parameters
""")
        with ac2:
            st.markdown("""
**QA Model — Dual Encoder**
- Embedding: Vocab × 96 dims
- BiLSTM: 2 layers, 128 hidden
- Self-attention pooling + Dense (128 → 64)
- Cosine similarity scoring, ~1.65M parameters
""")

    st.markdown("---")
    st.markdown("### Train the AI Model")
    st.markdown("""
    <div class="info-box">
        Upload transport law PDFs — Motor Vehicles Act, MoRTH reports, FAR/DAR form guides —
        and the AI will learn from them. Training is required only once.
        More documents improve accuracy.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    up1, up2 = st.columns([2, 1])
    with up1:
        uploaded = st.file_uploader(
            "Upload PDF documents (multiple files allowed):",
            type=["pdf"], accept_multiple_files=True
        )
    with up2:
        csv_file = st.file_uploader(
            "Accident news CSV (optional):", type=["csv"]
        )

    tc1, tc2 = st.columns(2)
    with tc1:
        ner_ep = st.slider("NER training epochs",    5, 60, 30,
                           help="Higher = more accurate but slower. 30 recommended.")
    with tc2:
        qa_ep  = st.slider("QA training epochs",     5, 40, 20)

    if uploaded:
        st.markdown(
            f'<div class="success-box">'
            f'<strong>{len(uploaded)} PDF(s) ready:</strong> '
            + ", ".join(f.name for f in uploaded[:5])
            + ("..." if len(uploaded) > 5 else "")
            + "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    if uploaded and st.button("Start Training", type="primary", use_container_width=False):
        os.makedirs("data/pdfs", exist_ok=True)
        os.makedirs("models",    exist_ok=True)
        for pdf in uploaded:
            with open(f"data/pdfs/{pdf.name}", "wb") as f:
                f.write(pdf.read())
        csv_arg = ""
        if csv_file:
            with open("data/accidents.csv", "wb") as f:
                f.write(csv_file.read())
            csv_arg = " --csv-file data/accidents.csv"

        app_dir = os.path.dirname(os.path.abspath(__file__))
        cmd = (f"cd {app_dir} && python pytorch_train.py "
               f"--pdf-dir data/pdfs --models-dir models "
               f"--ner-epochs {ner_ep} --qa-epochs {qa_ep}{csv_arg}")
        result_q = queue.Queue()

        def _run(c, q):
            import subprocess
            q.put(subprocess.run(c, shell=True, capture_output=True, text=True))

        t = threading.Thread(target=_run, args=(cmd, result_q), daemon=True)
        t.start()

        with st.status("Training in progress — please wait...", expanded=True) as status:
            ph = st.empty(); dots = 0
            while t.is_alive():
                dots = (dots % 4) + 1
                ph.markdown(f"Training{'.' * dots} Do not close this page.")
                time.sleep(3)
            ph.empty()
            res = result_q.get()
            if res.returncode == 0:
                status.update(label="Training complete. The AI models are ready.",
                              state="complete")
                st.balloons()
            else:
                status.update(label="Training failed. See error details below.", state="error")
                with st.expander("Error details"):
                    st.code(res.stderr[-600:])

        st.cache_resource.clear()
        st.rerun()

    elif not uploaded:
        st.markdown("""
        <div style="font-size:0.82rem;color:#888;margin-top:8px;">
            Upload at least one PDF document to enable training.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Reference Guide")

    with st.expander("How does this tool work?"):
        st.markdown("""
        This tool uses artificial intelligence trained on Indian road transport documents.

        1. Paste accident text (from news, FIR, press release) or upload a PDF
        2. The BiLSTM-CRF NER model extracts key entities: location, vehicles, casualties, IPC sections
        3. Regex patterns run in parallel as a fallback for structured fields (dates, numbers)
        4. Fields are pre-filled in the FAR/DAR form
        5. Review, correct if needed, and download as PDF, TXT, or JSON

        Training data: Motor Vehicles Act, National Highways Act, MoRTH reports, accident news CSV.
        """)

    with st.expander("What is a FAR form?"):
        st.markdown("""
        **FAR — First Accident Report (Form 1)**
        - Filed by the investigating police officer
        - Deadline: within 48 hours of the accident
        - Submitted to the Motor Accidents Claims Tribunal (MACT)
        - Contains: date, place, vehicles, casualties, cause, IPC sections charged
        """)

    with st.expander("What is a DAR form?"):
        st.markdown("""
        **DAR — Detailed Accident Report (Form VII)**
        - Filed by the investigating police officer
        - Deadline: within 90 days of the accident
        - More detailed than FAR — includes driver details, witness info, investigation findings
        - Also submitted to the Claims Tribunal
        """)

    with st.expander("About the AI models"):
        st.markdown("""
        **NER Model:** Uses a BiLSTM-CRF architecture — a recurrent neural network that reads
        text bidirectionally and uses a Conditional Random Field with Viterbi decoding to output
        a globally optimal entity tag sequence (BIO tagging scheme).

        **QA Model:** Uses a Dual Encoder / Siamese BiLSTM with self-attention pooling.
        Encodes questions and passages into dense vectors, then ranks by cosine similarity.
        A hardcoded knowledge base of transport law FAQs provides keyword-based fallback scoring.
        """)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="gov-footer">
    Road Accident Reporting Tool &nbsp;|&nbsp;
    Ministry of Road Transport &amp; Highways &nbsp;|&nbsp;
    Government of India &nbsp;|&nbsp; All rights reserved
</div>
""", unsafe_allow_html=True)
