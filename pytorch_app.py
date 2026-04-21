"""
Surface Transport SLM — Customer-Friendly Streamlit App
========================================================
Designed for police officers, RTO staff, and government clerks.
No technical jargon. Simple guided workflow.

Run: python -m streamlit run pytorch_app.py
"""

import streamlit as st
import os, sys, json, tempfile, re
import threading, queue, time

sys.path.insert(0, os.path.dirname(__file__))

from pytorch_ner import NERPredictor, auto_annotate, tokenize
from pytorch_qa import QAPredictor, FormGenerator
from pdf_processor import extract_text_from_pdf, clean_text

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Road Accident Assistant — MoRTH",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: Light, friendly, approachable ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Nunito+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
}

/* Clean white background */
.stApp { background: #f0f4f8; }
[data-testid="stSidebar"] { background: #1a237e !important; }

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 4px 20px rgba(26,35,126,0.25);
}
.top-banner .icon { font-size: 3.2rem; }
.top-banner h1 {
    font-family: 'Nunito', sans-serif;
    font-size: 1.9rem; font-weight: 800;
    color: white; margin: 0; letter-spacing: -0.3px;
}
.top-banner .sub {
    color: #c5cae9; margin: 5px 0 0 0; font-size: 0.95rem; font-weight: 500;
}
.top-banner .badges { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.15); color: white;
    padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}

/* ── Step cards (big clickable-looking cards) ── */
.step-card {
    background: white;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border: 2px solid #e8edf2;
    height: 100%;
    transition: border-color 0.2s;
}
.step-card:hover { border-color: #3949ab; }
.step-number {
    background: #3949ab; color: white;
    width: 32px; height: 32px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700; margin-bottom: 12px;
    font-family: 'Nunito', sans-serif;
}
.step-title {
    font-family: 'Nunito', sans-serif;
    font-size: 1.05rem; font-weight: 700; color: #1a237e; margin-bottom: 4px;
}
.step-desc { font-size: 0.85rem; color: #64748b; }

/* ── Big friendly tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 6px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #1a237e !important;
    color: white !important;
}

/* ── White content card ── */
.content-card {
    background: white;
    border-radius: 14px;
    padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}

/* ── Result chips for entities ── */
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px; border-radius: 20px; margin: 4px;
    font-size: 0.85rem; font-weight: 600;
}
.chip-icon { font-size: 0.9rem; }
.chip-red    { background: #fef2f2; color: #dc2626; border: 1.5px solid #fecaca; }
.chip-orange { background: #fff7ed; color: #ea580c; border: 1.5px solid #fed7aa; }
.chip-green  { background: #f0fdf4; color: #16a34a; border: 1.5px solid #bbf7d0; }
.chip-blue   { background: #eff6ff; color: #2563eb; border: 1.5px solid #bfdbfe; }
.chip-purple { background: #faf5ff; color: #7c3aed; border: 1.5px solid #ddd6fe; }
.chip-indigo { background: #eef2ff; color: #4338ca; border: 1.5px solid #c7d2fe; }
.chip-yellow { background: #fefce8; color: #ca8a04; border: 1.5px solid #fef08a; }
.chip-teal   { background: #f0fdfa; color: #0d9488; border: 1.5px solid #99f6e4; }
.chip-gray   { background: #f8fafc; color: #475569; border: 1.5px solid #e2e8f0; }

/* ── Big stat boxes ── */
.stat-box {
    background: white;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.stat-box .num {
    font-family: 'Nunito', sans-serif;
    font-size: 2.2rem; font-weight: 800; line-height: 1;
}
.stat-box .lbl {
    font-size: 0.78rem; font-weight: 600; margin-top: 6px;
    text-transform: uppercase; letter-spacing: 0.5px; color: #64748b;
}
.stat-red    .num { color: #dc2626; }
.stat-orange .num { color: #ea580c; }
.stat-blue   .num { color: #2563eb; }
.stat-green  .num { color: #16a34a; }

/* ── FAR field rows ── */
.far-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #f1f5f9;
}
.far-row:last-child { border-bottom: none; }
.far-row .far-icon { font-size: 1.2rem; margin-top: 2px; }
.far-row .far-lbl  { font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.5px; color: #94a3b8; }
.far-row .far-val  { font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-top: 2px; }

/* ── Result answer card ── */
.answer-card {
    background: white;
    border-radius: 12px;
    padding: 20px 22px;
    border-left: 4px solid #3949ab;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.answer-card .ans-num {
    font-family: 'Nunito', sans-serif;
    font-size: 0.75rem; font-weight: 700; color: #3949ab;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.answer-card .ans-text { font-size: 0.92rem; color: #334155; line-height: 1.75; }
.answer-card .ans-src  { font-size: 0.75rem; color: #94a3b8; margin-top: 10px; }

/* ── Info box ── */
.info-box {
    background: #eff6ff; border: 1.5px solid #bfdbfe; border-radius: 10px;
    padding: 14px 18px; font-size: 0.87rem; color: #1e40af;
}
.warn-box {
    background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 10px;
    padding: 14px 18px; font-size: 0.87rem; color: #92400e;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a237e, #3949ab) !important;
    border: none !important; color: white !important;
    padding: 12px 28px !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 12px rgba(26,35,126,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: white !important;
    border: 2px solid #e2e8f0 !important;
    color: #475569 !important;
}

/* ── Text inputs ── */
.stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    font-family: 'Nunito Sans', sans-serif !important;
    font-size: 0.92rem !important;
    background: #fafbfc !important;
    color: #1e293b !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #3949ab !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #f0fdf4 !important;
    border: 2px solid #bbf7d0 !important;
    color: #16a34a !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Radio ── */
.stRadio > div { gap: 12px !important; }
.stRadio label { font-weight: 600 !important; font-size: 0.9rem !important; }

/* Divider */
hr { border-color: #e2e8f0 !important; }

/* Force dark text for all labels and content in main area */
.stApp label, .stApp .stRadio label, .stApp .stMarkdown,
.stApp p, .stApp span, .stApp .stTextArea label,
.stApp .stTextInput label, .stApp .stRadio div {
    color: #1e293b !important;
}
.stApp .stExpander summary span { color: #1e293b !important; }
/* Keep banner white text */
.top-banner h1, .top-banner .sub, .top-banner span,
.badge, .top-banner p { color: inherit !important; }
.top-banner h1 { color: white !important; }
.top-banner .sub { color: #c5cae9 !important; }

/* Section heading */
.sec-head {
    font-family: 'Nunito', sans-serif;
    font-size: 1.0rem; font-weight: 800; color: #1a237e;
    margin: 0 0 16px 0;
    display: flex; align-items: center; gap: 8px;
}

/* Empty state */
.empty-state {
    background: #f8fafc; border: 2px dashed #cbd5e1;
    border-radius: 14px; padding: 52px 30px; text-align: center;
}
.empty-state .e-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state .e-title {
    font-family: 'Nunito', sans-serif;
    font-size: 1.05rem; font-weight: 700; color: #475569; margin-bottom: 6px;
}
.empty-state .e-sub  { font-size: 0.85rem; color: #94a3b8; }
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

# ── Top Banner ────────────────────────────────────────────────
ner_status = "✅ AI Ready" if ner_ready else "⚠️ Not Trained"
qa_status  = "✅ QA Ready" if qa_ready  else "⚠️ Not Trained"

st.markdown(f"""
<div class="top-banner">
    <div class="icon">🚦</div>
    <div>
        <h1>Road Accident Assistant</h1>
        <p class="sub">Ministry of Road Transport &amp; Highways — Automated Accident Reporting Tool</p>
        <div class="badges">
            <span class="badge">🤖 {ner_status}</span>
            <span class="badge">📖 {qa_status}</span>
            <span class="badge">🇮🇳 MoRTH</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────
if not ner_ready:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <strong>AI model not trained yet.</strong>
        Go to the <strong>⚙️ Settings</strong> tab to upload PDFs and train the model first.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
for col, num, icon, title, desc in [
    (c1, "1", "📝", "Paste Accident News", "Copy any accident report or news article and paste it in the tool"),
    (c2, "2", "🤖", "AI Reads the Text",   "The AI automatically finds key details — who, where, what happened"),
    (c3, "3", "📋", "Get Report Form",     "Download a pre-filled FAR/DAR accident report — ready to submit"),
]:
    with col:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-title">{icon} {title}</div>
            <div class="step-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📝  Fill Accident Report",
    "❓  Ask a Question",
    "📄  Blank Forms",
    "⚙️  Settings & Training",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — FILL ACCIDENT REPORT (main feature)
# ═══════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="sec-head">📰 Paste Your Accident Text</div>',
                    unsafe_allow_html=True)

        mode = st.radio("How do you want to provide the accident information?",
                        ["📋  Paste or type text", "📄  Upload a PDF file"],
                        label_visibility="collapsed")

        if "Paste" in mode:
            sample = (
                "Five persons died and about three dozen were injured in a "
                "head-on collision of a passenger bus and a truck near "
                "Mathania in Jodhpur on Friday afternoon. The accident "
                "occurred due to overspeeding on State Highway 61 in "
                "Rajasthan. Police registered a case under Section 304A "
                "and Section 279 of the IPC. "
                "The truck registration was RJ 14 GA 1234."
            )
            text = st.text_area(
                "Paste the accident news or report text here:",
                value=sample, height=260,
                help="You can paste from any news article, FIR, or press release"
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
                with st.spinner("Reading PDF…"):
                    res  = extract_text_from_pdf(tmp.name)
                    text = clean_text(res.get("full_text", ""))
                os.unlink(tmp.name)
                st.success(f"✅ PDF read successfully — {len(text)} characters extracted")
                st.text_area("Extracted text (preview):", value=text[:1500],
                             height=200, disabled=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🤖  Analyse Text & Fill Report",
                            type="primary", use_container_width=True,
                            disabled=not ner_ready)
        if not ner_ready:
            st.markdown("""
            <div style="font-size:0.82rem;color:#94a3b8;margin-top:8px;text-align:center;">
                ⚠️ AI not trained yet. Go to ⚙️ Settings tab to train first.
            </div>""", unsafe_allow_html=True)

        # ── Sample texts ──
        with st.expander("📌 Try a sample accident text"):
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

    # ── RIGHT: Results ──
    with right:
        if run_btn and text:
            with st.spinner("🤖 AI is reading the text…"):
                if ner_ready:
                    entities = ner_pred.extract_entities(text)
                    far      = ner_pred.extract_to_far_format(text)
                else:
                    annotated = auto_annotate(text)
                    entities  = {}
                    for tok, tag in annotated:
                        if tag != "O":
                            etype = tag.split("-", 1)[1]
                            entities.setdefault(etype, [])
                            if tag.startswith("B-"): entities[etype].append(tok)
                            elif entities[etype]:   entities[etype][-1] += " " + tok
                    far = {}

            # ── Quick stats ──
            killed  = far.get("number_of_fatalities", "0") or "0"
            injured = far.get("number_of_injured",    "0") or "0"
            loc     = far.get("place_of_accident",    "—") or "—"
            nature  = far.get("nature_of_accident",   "—") or "—"

            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, val, lbl, cls in [
                (sc1, killed,  "Persons Killed",  "stat-red"),
                (sc2, injured, "Persons Injured", "stat-orange"),
                (sc3, loc[:8]+"…" if len(loc)>8 else loc, "Location", "stat-blue"),
                (sc4, nature,  "Nature",          "stat-green"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="stat-box {cls}">
                        <div class="num">{val}</div>
                        <div class="lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── What the AI found ──
            if entities:
                st.markdown('<div class="sec-head">🔍 What the AI Found</div>',
                            unsafe_allow_html=True)

                FRIENDLY = {
                    "NUM_KILLED":   ("💀", "Persons Killed",  "chip-red"),
                    "NUM_INJURED":  ("🏥", "Persons Injured", "chip-orange"),
                    "LOCATION":     ("📍", "Place",           "chip-blue"),
                    "STATE":        ("🗺️", "State",           "chip-green"),
                    "VEHICLE":      ("🚗", "Vehicles",        "chip-indigo"),
                    "COLLISION":    ("💥", "Collision Type",  "chip-purple"),
                    "CAUSE":        ("⚠️", "Cause",           "chip-yellow"),
                    "ROAD_TYPE":    ("🛣️", "Road",            "chip-teal"),
                    "DATE":         ("📅", "Date",            "chip-gray"),
                    "TIME":         ("🕐", "Time",            "chip-gray"),
                    "WEATHER":      ("🌧️", "Weather",         "chip-blue"),
                    "IPC_SECTION":  ("⚖️", "IPC Section",     "chip-gray"),
                    "REG_NUMBER":   ("🔢", "Vehicle Reg No.", "chip-green"),
                    "VICTIM_TYPE":  ("👤", "Victim Type",     "chip-gray"),
                    "SEVERITY":     ("🚨", "Severity",        "chip-red"),
                }
                ORDER = ["NUM_KILLED","NUM_INJURED","LOCATION","STATE","VEHICLE",
                         "COLLISION","CAUSE","ROAD_TYPE","DATE","TIME",
                         "IPC_SECTION","REG_NUMBER","WEATHER","SEVERITY","VICTIM_TYPE"]

                chips_html = ""
                for etype in ORDER:
                    if etype not in entities: continue
                    icon, label, css = FRIENDLY.get(etype, ("•", etype, "chip-gray"))
                    for v in entities[etype]:
                        chips_html += (f'<span class="chip {css}">'
                                       f'<span class="chip-icon">{icon}</span>'
                                       f'{label}: <strong>{v}</strong></span>')

                st.markdown(f'<div style="line-height:2.8;">{chips_html}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warn-box">
                    ⚠️ The AI could not find specific accident details in this text.
                    Make sure the text mentions location, vehicles, or casualties.
                </div>""", unsafe_allow_html=True)

            # ── Pre-filled FAR form ──
            if far and any(v for v in far.values()):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-head">📋 Pre-filled Accident Report (FAR)</div>',
                            unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    ℹ️ These fields have been automatically filled from the text.
                    Please review and correct if needed before downloading.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                FAR_DISPLAY = [
                    ("📅", "Date of Accident",   far.get("date_of_accident")),
                    ("🕐", "Time of Accident",   far.get("time_of_accident")),
                    ("📍", "Place of Accident",  far.get("place_of_accident")),
                    ("🗺️", "State",              far.get("state")),
                    ("🚨", "Nature of Accident", far.get("nature_of_accident")),
                    ("🚗", "Vehicle 1",          far.get("vehicle_1")),
                    ("🚛", "Vehicle 2",          far.get("vehicle_2")),
                    ("💀", "Persons Killed",     far.get("number_of_fatalities")),
                    ("🏥", "Persons Injured",    far.get("number_of_injured")),
                    ("🛣️", "Road Type",          far.get("road_type")),
                    ("💥", "Collision Type",     far.get("collision_type")),
                    ("⚠️", "Cause of Accident",  far.get("cause_of_accident")),
                    ("🔢", "Reg. Number",        far.get("registration_number")),
                    ("🌧️", "Weather",            far.get("weather_condition")),
                    ("⚖️", "IPC Sections",       far.get("ipc_sections")),
                ]
                rows_html = ""
                for icon, label, value in FAR_DISPLAY:
                    if value and value not in ("0", ""):
                        rows_html += f"""
                        <div class="far-row">
                            <div class="far-icon">{icon}</div>
                            <div>
                                <div class="far-lbl">{label}</div>
                                <div class="far-val">{value}</div>
                            </div>
                        </div>"""

                st.markdown(f'<div class="content-card">{rows_html}</div>',
                            unsafe_allow_html=True)

                # Download
                fg        = FormGenerator()
                form      = fg.prefill_far(far) if ner_ready else fg.get_far_template()
                form_text = fg.form_to_text(form)
                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("📥  Download FAR (.txt)",
                                       data=form_text, file_name="FAR_draft.txt",
                                       use_container_width=True)
                with dl2:
                    st.download_button("📥  Download as JSON",
                                       data=json.dumps(far, indent=2),
                                       file_name="FAR_data.json",
                                       use_container_width=True)

        elif run_btn and not text:
            st.warning("⚠️ Please paste some text or upload a PDF first.")

        else:
            st.markdown("""
            <div class="empty-state">
                <div class="e-icon">📋</div>
                <div class="e-title">Your report will appear here</div>
                <div class="e-sub">Paste accident text on the left<br>
                    and click the blue button to analyse it</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — ASK A QUESTION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-head">❓ Ask About Traffic Rules & Regulations</div>',
                unsafe_allow_html=True)

    if not qa_ready:
        st.markdown("""
        <div class="warn-box">
            ⚠️ This feature requires training first. Go to the <strong>⚙️ Settings</strong> tab.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            💡 Ask any question about the Motor Vehicles Act, accident reporting rules,
            road safety guidelines, or transport regulations.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        q = st.text_input("Type your question here:",
                          placeholder="e.g. What is the time limit to file a First Accident Report?",
                          label_visibility="collapsed")

        # Friendly sample questions
        st.markdown("**Or click a common question:**")
        sample_qs = [
            "What is a fatal accident?",
            "What is the time limit for filing a FAR report?",
            "Who is responsible for filing the accident report?",
            "What information is required in the accident report?",
            "What is the Motor Vehicles Act?",
            "What are the rules for hit and run accidents?",
        ]
        q_cols = st.columns(3)
        for i, sq in enumerate(sample_qs):
            with q_cols[i % 3]:
                if st.button(sq, key=f"q_{i}", use_container_width=True):
                    q = sq

        if q:
            with st.spinner("🔍 Searching regulations…"):
                results = qa_pred.answer(q, top_k=3)

            st.markdown(f"<br>**Results for:** *{q}*", unsafe_allow_html=True)
            for i, r in enumerate(results):
                st.markdown(f"""
                <div class="answer-card">
                    <div class="ans-num">Answer {i+1} of 3</div>
                    <div class="ans-text">{r['answer']}</div>
                    <div class="ans-src">📄 Source: {r['source']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:0.8rem;color:#94a3b8;margin-top:12px;">
                ⚠️ These answers are extracted from uploaded transport law documents.
                Always verify with official sources before taking legal action.
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — BLANK FORMS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">📄 Download Blank Accident Report Forms</div>',
                unsafe_allow_html=True)

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:2rem;margin-bottom:10px;">📋</div>
            <div style="font-family:'Nunito',sans-serif;font-size:1.05rem;
                        font-weight:700;color:#1a237e;margin-bottom:6px;">
                First Accident Report (FAR)
            </div>
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">
                Must be filed by investigating officer within <strong>48 hours</strong>
                of the accident. Submitted to the Claims Tribunal.
            </div>
        </div>
        """, unsafe_allow_html=True)

        autofill_1 = st.text_area("Auto-fill from accident text (optional):",
                                  height=100, key="far_fill",
                                  placeholder="Paste accident text here to auto-fill fields…")
        if st.button("📋  Generate FAR Form", type="primary",
                     use_container_width=True, key="gen_far"):
            fg  = FormGenerator()
            ner_local, ner_ok2 = load_ner()
            if autofill_1 and ner_ok2:
                ext  = ner_local.extract_to_far_format(autofill_1)
                form = fg.prefill_far(ext)
            else:
                form = fg.get_far_template()

            st.markdown(f"**{form['form_title']}**")
            vals = {}
            for field in form["fields"]:
                if field["type"] == "select":
                    opts = field.get("options", [])
                    idx  = opts.index(field["value"]) if field["value"] in opts else 0
                    vals[field["id"]] = st.selectbox(field["label"], opts, index=idx,
                                                     key=f"far_{field['id']}")
                elif field["type"] == "number":
                    try: v = int(field.get("value","") or 0)
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
            lines = [form["form_title"], "="*50]
            for field in form["fields"]:
                lines.append(f"{field['label']}: {vals.get(field['id'],'')}")
            st.download_button("📥  Download Filled FAR",
                               data="\n".join(lines), file_name="FAR_filled.txt",
                               use_container_width=True)

    with f2:
        st.markdown("""
        <div class="content-card">
            <div style="font-size:2rem;margin-bottom:10px;">📑</div>
            <div style="font-family:'Nunito',sans-serif;font-size:1.05rem;
                        font-weight:700;color:#1a237e;margin-bottom:6px;">
                Detailed Accident Report (DAR)
            </div>
            <div style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">
                Must be filed by investigating officer within <strong>90 days</strong>
                of the accident. Contains complete investigation details.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📑  Generate DAR Form", type="primary",
                     use_container_width=True, key="gen_dar"):
            fg   = FormGenerator()
            form = fg.get_dar_template()
            st.markdown(f"**{form['form_title']}**")
            vals = {}
            for field in form["fields"]:
                if field["type"] == "select":
                    opts = field.get("options", [])
                    idx  = opts.index(field["value"]) if field["value"] in opts else 0
                    vals[field["id"]] = st.selectbox(field["label"], opts, index=idx,
                                                     key=f"dar_{field['id']}")
                elif field["type"] == "number":
                    try: v = int(field.get("value","") or 0)
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
            lines = [form["form_title"], "="*50]
            for field in form["fields"]:
                lines.append(f"{field['label']}: {vals.get(field['id'],'')}")
            st.download_button("📥  Download Filled DAR",
                               data="\n".join(lines), file_name="DAR_filled.txt",
                               use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SETTINGS & TRAINING
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">⚙️ System Status & Training</div>',
                unsafe_allow_html=True)

    # Status
    s1, s2 = st.columns(2)
    with s1:
        color = "#f0fdf4" if ner_ready else "#fffbeb"
        bcolor = "#bbf7d0" if ner_ready else "#fde68a"
        icon  = "✅" if ner_ready else "⚠️"
        label = "AI Model (Accident Analyser)" + (" — Ready" if ner_ready else " — Not Trained")
        st.markdown(f"""
        <div style="background:{color};border:2px solid {bcolor};border-radius:12px;
                    padding:18px 20px;margin-bottom:12px;">
            <div style="font-weight:700;font-size:0.95rem;">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)
        if ner_ready:
            try:
                import torch
                ck = torch.load("models/ner_bilstm_crf.pt", map_location="cpu", weights_only=False)
                st.caption(f"Model size: {ck['total_params']:,} parameters")
            except: pass

    with s2:
        color = "#f0fdf4" if qa_ready else "#fffbeb"
        bcolor = "#bbf7d0" if qa_ready else "#fde68a"
        icon  = "✅" if qa_ready else "⚠️"
        label = "Q&A Model (Regulations)" + (" — Ready" if qa_ready else " — Not Trained")
        st.markdown(f"""
        <div style="background:{color};border:2px solid {bcolor};border-radius:12px;
                    padding:18px 20px;margin-bottom:12px;">
            <div style="font-weight:700;font-size:0.95rem;">{icon} {label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏋️ Train the AI Model")
    st.markdown("""
    <div class="info-box">
        Upload your transport law PDFs (Motor Vehicles Act, MoRTH reports, FAR/DAR forms, etc.)
        and the AI will learn from them. This only needs to be done once.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF documents (you can select multiple):",
                                type=["pdf"], accept_multiple_files=True)
    csv_file = st.file_uploader("Upload accident news CSV file (optional — improves accuracy):",
                                type=["csv"])

    tc1, tc2 = st.columns(2)
    with tc1:
        ner_ep = st.slider("Training intensity (more = better but slower)",
                           5, 60, 30,
                           help="Higher = more training cycles = more accurate but takes longer")
    with tc2:
        qa_ep = st.slider("Q&A training intensity", 5, 40, 20)

    if uploaded:
        st.markdown(f"**{len(uploaded)} PDF(s) selected:** " +
                    ", ".join(f.name for f in uploaded[:5]) +
                    ("…" if len(uploaded) > 5 else ""))

    if uploaded and st.button("🚀  Start Training", type="primary", use_container_width=False):
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

        with st.status("🚀 Training in progress… this may take a few minutes",
                       expanded=True) as status:
            ph = st.empty(); dots = 0
            while t.is_alive():
                dots = (dots % 4) + 1
                ph.markdown(f"⏳ Training{'.' * dots} Please wait, do not close this page.")
                time.sleep(3)
            ph.empty()
            res = result_q.get()
            if res.returncode == 0:
                status.update(label="✅ Training complete! The AI is ready to use.",
                              state="complete")
                st.balloons()
            else:
                status.update(label="❌ Training failed. See error below.", state="error")
                with st.expander("Error details"):
                    st.code(res.stderr[-600:])

        st.cache_resource.clear()
        st.rerun()

    elif not uploaded:
        st.markdown("""
        <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px;">
            👆 Upload at least one PDF to enable training.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    with st.expander("How does this tool work?"):
        st.markdown("""
        This tool uses **Artificial Intelligence** trained on Indian road transport documents
        to help fill accident report forms automatically.

        1. **You paste** accident text (from news, FIR, press release)
        2. **The AI reads** the text and identifies: who was killed/injured, where it happened,
           which vehicles were involved, what caused it, IPC sections, etc.
        3. **The tool fills** the FAR/DAR form fields for you automatically
        4. **You review** and download the pre-filled form

        The AI has been trained on: Motor Vehicles Act, National Highways Act,
        IRC accident recording forms, and real accident news articles.
        """)
    with st.expander("What is a FAR form?"):
        st.markdown("""
        **FAR = First Accident Report**

        - Must be filed by the **investigating police officer**
        - Deadline: **within 48 hours** of the accident
        - Submitted to: **Motor Accidents Claims Tribunal (MACT)**
        - Contains: date, place, vehicles involved, casualties, cause, IPC sections
        """)
    with st.expander("What is a DAR form?"):
        st.markdown("""
        **DAR = Detailed Accident Report**

        - Must be filed by the **investigating police officer**
        - Deadline: **within 90 days** of the accident
        - More detailed than FAR — includes driver details, witness info, full investigation
        - Also submitted to the Claims Tribunal
        """)