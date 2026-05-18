import streamlit as st
import joblib, os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Aging Risk Analyzer",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d0f14 0%, #111520 60%, #0d0f14 100%);
}
[data-testid="stSidebar"] {
    background: #13161f !important;
    border-right: 1px solid #1e2235;
}
[data-testid="stSidebar"] label {
    color: #7a8099 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

/* ── Sidebar sections ── */
.sidebar-section {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #2d3250;
    margin: 1.5rem 0 0.6rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2235;
}

/* ── Hero ── */
.hero-eyebrow {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.22em;
    text-transform: uppercase; color: #c0392b; margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem; color: #f0f2f8; line-height: 1.08; margin-bottom: 0.4rem;
}
.hero-sub {
    font-size: 0.88rem; color: #4a5070; line-height: 1.7; max-width: 560px;
}

/* ── Divider ── */
.hr { border: none; border-top: 1px solid #1a1d2e; margin: 1.6rem 0; }

/* ── Risk card ── */
.risk-card {
    border-radius: 18px; padding: 2.2rem 2.4rem;
    text-align: center; margin-bottom: 1.2rem;
}
.risk-card-high {
    background: linear-gradient(135deg, #2a0f0f 0%, #1a0808 100%);
    border: 1px solid #7f1d1d;
    box-shadow: 0 0 50px rgba(185, 28, 28, 0.2);
}
.risk-card-low {
    background: linear-gradient(135deg, #0a1f12 0%, #061510 100%);
    border: 1px solid #14532d;
    box-shadow: 0 0 50px rgba(21, 128, 61, 0.18);
}
.risk-eyebrow {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.22em;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.risk-eyebrow-high { color: #ef4444; }
.risk-eyebrow-low  { color: #22c55e; }
.risk-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem; line-height: 1; margin-bottom: 0.5rem;
}
.risk-verdict-high { color: #fca5a5; }
.risk-verdict-low  { color: #86efac; }
.risk-prob { font-size: 0.85rem; color: #6b7280; letter-spacing: 0.03em; }

/* ── Stat pills ── */
.stat-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.stat-pill {
    background: #13161f; border: 1px solid #1e2235; border-radius: 8px;
    padding: 0.5rem 0.9rem; font-size: 0.76rem; color: #9ca3af;
}
.stat-pill span { color: #e2e8f0; font-weight: 600; margin-left: 0.3rem; }

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #c0392b, #922b21) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.85rem 1.2rem !important;
    font-size: 0.82rem !important; font-weight: 700 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    margin-top: 1rem;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Info / sidebar cards ── */
.info-box {
    background: #13161f; border: 1px solid #1e2235;
    border-left: 3px solid #c0392b; border-radius: 8px;
    padding: 1rem 1.2rem; font-size: 0.81rem; color: #6b7280; line-height: 1.75;
}

/* ── Explain cards ── */
.explain-card {
    background: #13161f; border: 1px solid #1e2235;
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
}
.explain-header {
    display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;
}
.explain-title {
    font-size: 0.82rem; font-weight: 600; color: #e2e8f0; letter-spacing: 0.03em;
}
.explain-body { font-size: 0.81rem; color: #6b7280; line-height: 1.75; }

/* ── Recommendation cards ── */
.rec-card {
    background: #0a1a10; border: 1px solid #163320;
    border-left: 3px solid #22c55e; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    font-size: 0.82rem; color: #86efac; line-height: 1.65;
}

/* ── Bio-age score ── */
.bio-wrap {
    text-align: center; padding: 1.8rem 1.4rem;
    background: #13161f; border: 1px solid #1e2235; border-radius: 16px;
    margin-top: 0.8rem;
}
.bio-num {
    font-family: 'DM Serif Display', serif; font-size: 3.4rem; line-height: 1;
}
.bio-tag {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; margin-top: 0.4rem;
}
.bio-bar-track {
    background: #1e2235; border-radius: 8px; height: 8px;
    margin: 0.8rem auto 1rem auto; max-width: 280px;
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #3d4260; margin-bottom: 0.8rem;
}

/* ── Footer ── */
.footer {
    text-align: center; font-size: 0.7rem; color: #252840;
    letter-spacing: 0.1em; margin-top: 3rem; padding-top: 1.6rem;
    border-top: 1px solid #161929;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'age','sex','cp','trestbps','chol','fbs',
    'restecg','thalach','exang','oldpeak','slope','ca','thal'
]

FEATURE_LABELS = {
    'age':      'Age',
    'sex':      'Biological Sex',
    'cp':       'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol':     'Cholesterol',
    'fbs':      'Fasting Blood Sugar',
    'restecg':  'Resting ECG',
    'thalach':  'Max Heart Rate',
    'exang':    'Exercise Angina',
    'oldpeak':  'ST Depression',
    'slope':    'ST Slope',
    'ca':       'Major Vessels',
    'thal':     'Perfusion Type',
}

CP_LABELS      = {0:'Typical Angina', 1:'Atypical Angina', 2:'Non-Anginal Pain', 3:'Asymptomatic'}
THAL_LABELS    = {0:'Normal', 1:'Fixed Defect', 2:'Reversible Defect', 3:'Unknown'}
SLOPE_LABELS   = {0:'Upsloping', 1:'Flat', 2:'Downsloping'}
RESTECG_LABELS = {0:'Normal', 1:'ST-T Abnormality', 2:'LV Hypertrophy'}

PLAIN_ENGLISH = {
    'thalach': {
        'icon':'💓', 'threshold':140, 'direction':'low_is_bad',
        'title':'Max heart rate — {val} bpm',
        'bad': 'Lower than expected for your age. Your heart reaches its ceiling sooner during exercise, suggesting reduced cardiovascular reserve — one of the clearest functional signs of cardiovascular aging.',
        'good':'A strong result. High max heart rate means your heart still has plenty of capacity, which is one of the best available proxies for a biologically younger cardiovascular system.',
    },
    'ca': {
        'icon':'🩻', 'threshold':1, 'direction':'high_is_bad',
        'title':'Major vessels with blockage — {val}',
        'bad': 'Fatty plaque has built up in one or more major arteries — think of rust accumulating inside a pipe over decades. Each additional blocked vessel increases the workload on your heart.',
        'good':'No significant vessel blockage detected. Clear arteries are among the strongest protective factors against heart disease.',
    },
    'oldpeak': {
        'icon':'📉', 'threshold':1.5, 'direction':'high_is_bad',
        'title':'ST depression score — {val}',
        'bad': "During exercise, your heart's electrical signal dips more than expected. This suggests that parts of your heart muscle may not be receiving enough oxygen under stress — a sign of ischemic strain.",
        'good':"Your heart's electrical signal stays stable under exercise load — a good indicator that your myocardium is getting adequate blood supply even when demand increases.",
    },
    'cp': {
        'icon':'🫀', 'threshold':1, 'direction':'high_is_bad',
        'title':'Chest pain pattern — {val_label}',
        'bad': 'Atypical or absent chest pain is associated with higher disease risk in this dataset. Atypical patterns become more common with age as nerve sensitivity changes.',
        'good':'Typical angina is a well-characterised pattern that is generally easier to diagnose and manage.',
    },
    'thal': {
        'icon':'🔬', 'threshold':1, 'direction':'high_is_bad',
        'title':'Heart perfusion type — {val_label}',
        'bad': 'An abnormal perfusion pattern indicates either scar tissue from a prior event (fixed defect) or living heart muscle that struggles under stress (reversible defect).',
        'good':'Normal blood flow pattern through the heart muscle. Your myocardium is receiving good circulation.',
    },
    'slope': {
        'icon':'📊', 'threshold':1, 'direction':'high_is_bad',
        'title':'ST slope during exercise — {val_label}',
        'bad': 'A flat or downsloping ST segment during peak exercise suggests stiffer arteries with less capacity to dilate on demand — a hallmark of vascular aging.',
        'good':'An upsloping ST pattern is the healthy response to exercise — your coronary arteries are flexible and responsive.',
    },
    'trestbps': {
        'icon':'🩺', 'threshold':140, 'direction':'high_is_bad',
        'title':'Resting blood pressure — {val} mmHg',
        'bad': 'Elevated even at rest. Think of it as your heart permanently pushing fluid through a narrower hose — increased mechanical stress accelerates arterial aging.',
        'good':'Blood pressure in a healthy range at rest means your heart and arteries are not under chronic mechanical strain.',
    },
    'chol': {
        'icon':'🧪', 'threshold':240, 'direction':'high_is_bad',
        'title':'Cholesterol — {val} mg/dl',
        'bad': 'Higher circulating cholesterol means more substrate for plaque formation on artery walls — a slow, decades-long process that is a core driver of cardiovascular aging.',
        'good':'Cholesterol in a manageable range reduces the rate of arterial plaque accumulation.',
    },
    'age': {
        'icon':'📅', 'threshold':55, 'direction':'high_is_bad',
        'title':'Chronological age — {val} years',
        'bad': 'Age increases baseline risk, but this model consistently ranks functional markers higher than the calendar. How your heart performs matters more than when you were born.',
        'good':'You are in a lower-risk age bracket. Your functional biomarkers still carry more predictive weight than your age alone.',
    },
    'exang': {
        'icon':'🏃', 'threshold':0.5, 'direction':'high_is_bad',
        'title':'Exercise-induced chest pain — {val_label}',
        'bad': 'Chest discomfort during exercise signals a supply-demand mismatch in your coronary circulation — your heart is asking for more oxygen than it can receive under load.',
        'good':'No chest pain during exercise. Your heart handles increased physical demand without distress signals.',
    },
    'fbs': {
        'icon':'🍬', 'threshold':0.5, 'direction':'high_is_bad',
        'title':'Fasting blood sugar > 120 mg/dl — {val_label}',
        'bad': 'Elevated fasting glucose slowly damages the inner lining of blood vessels, accelerating arterial aging even years before a clinical diabetes diagnosis.',
        'good':'Normal fasting glucose. Good blood sugar regulation protects vascular walls from glycation damage.',
    },
    'restecg': {
        'icon':'⚡', 'threshold':0.5, 'direction':'high_is_bad',
        'title':'Resting ECG — {val_label}',
        'bad': 'An abnormal resting ECG suggests the heart\'s electrical conduction system has accumulated wear — often the result of long-term cardiovascular stress.',
        'good':'Normal resting electrical activity. Your heart\'s conduction system appears to be aging well.',
    },
    'sex': {
        'icon':'👤', 'threshold':0.5, 'direction':'high_is_bad',
        'title':'Biological sex — {val_label}',
        'bad': 'Men develop cardiovascular disease roughly 10 years earlier on average — not inevitable, but it makes earlier and more frequent screening valuable.',
        'good':'Women generally benefit from hormonal protection before menopause, resulting in a later average onset of cardiovascular disease.',
    },
}

RECOMMENDATIONS = {
    'thalach': {
        'bad': [
            '🏃 Aim for 150+ minutes of moderate cardio per week — brisk walking, cycling, or swimming. Even 20 minutes daily improves max heart rate measurably within 6–8 weeks.',
            '📈 Monitor heart rate during workouts and gradually extend your comfortable upper range over months. A fitness tracker helps track progress.',
            '😴 Prioritise 7–8 hours of sleep nightly — sleep deprivation directly suppresses cardiovascular fitness and cardiac recovery.',
        ],
        'good': [
            '✅ Your cardiac reserve is strong — keep your current activity level.',
            '💪 Add resistance training twice a week to complement aerobic fitness and preserve muscle mass, which declines with age.',
            '📊 Retest your cardio fitness every 6 months to track biological age trends over time.',
        ],
    },
    'ca': {
        'bad': [
            '🥗 Adopt a Mediterranean-style diet — the best-evidenced dietary pattern for slowing atherosclerosis progression in clinical trials.',
            '🚭 If you smoke, cessation is the single highest-impact intervention for vessel health — benefits begin within weeks.',
            '💊 Discuss statin therapy with your physician. Modern statins do more than lower LDL — they can stabilise existing plaques.',
        ],
        'good': [
            '✅ Clear arteries are a strong protective asset. Protect them with a low-saturated-fat, high-fibre diet.',
            '🫐 Eat antioxidant-rich foods (berries, leafy greens, olive oil) to reduce oxidative stress on vessel walls.',
            '🩺 Annual lipid panel to detect any trajectory changes early, before they progress.',
        ],
    },
    'oldpeak': {
        'bad': [
            '🧘 Manage psychological stress actively — elevated cortisol worsens myocardial oxygen efficiency and promotes inflammation.',
            '🩺 Ask your cardiologist about a stress echocardiogram for a detailed picture of how your heart performs under controlled load.',
            '💊 If exercise reliably triggers symptoms, discuss beta-blocker or nitrate therapy with your doctor.',
        ],
        'good': [
            '✅ Stable ST under exercise load is a good sign — your heart handles stress without electrical distress.',
            '🏊 Continue regular aerobic exercise; it maintains and actively improves myocardial oxygen efficiency.',
            '📉 Monitor blood pressure regularly — hypertension is the most common driver of ST changes over time.',
        ],
    },
    'trestbps': {
        'bad': [
            '🧂 Reduce dietary sodium to under 2 g/day — the fastest single dietary change for lowering blood pressure.',
            '🧘 Practice daily breathing exercises (4-7-8 breathing or box breathing) — clinically shown to lower systolic BP by 4–8 mmHg.',
            '🏃 30 minutes of moderate walking daily lowers systolic blood pressure by an average of 4–9 mmHg within weeks.',
        ],
        'good': [
            '✅ Healthy resting BP significantly reduces long-term arterial wear.',
            '🥑 Maintain potassium-rich foods (avocado, leafy greens, legumes) to support stable blood pressure.',
            '📊 Check blood pressure monthly — it often rises silently over months before symptoms appear.',
        ],
    },
    'chol': {
        'bad': [
            '🐟 Eat fatty fish (salmon, sardines, mackerel) twice a week — omega-3 fatty acids reduce triglycerides and vascular inflammation.',
            '🌾 Add soluble fibre (oats, psyllium, lentils) — it binds cholesterol in the gut before it reaches the bloodstream.',
            '🏋️ Strength training raises HDL (protective) cholesterol more effectively than cardio alone.',
        ],
        'good': [
            '✅ Cholesterol is well managed. Maintain with a diet rich in unsaturated fats.',
            '🫒 Use extra-virgin olive oil as your primary cooking fat to maintain healthy lipid ratios.',
            '🩺 Annual lipid panel — track LDL/HDL ratio, not just total cholesterol.',
        ],
    },
    'default': {
        'bad': [
            '🩺 Schedule a comprehensive cardiovascular checkup with your physician.',
            '📱 Track key biomarkers (resting heart rate, blood pressure, weight) weekly with a health app.',
            '🧬 Consider a longevity-focused blood panel to build a complete biological age picture.',
        ],
        'good': [
            '✅ Keep monitoring your biomarkers on a regular schedule.',
            '😴 Prioritise 7–8 hours of quality sleep — the most underrated cardiovascular intervention available.',
            '🤝 Maintain strong social connections — loneliness increases cardiovascular risk as much as smoking 15 cigarettes per day.',
        ],
    },
}


def get_val_label(feature, val):
    if feature == 'cp':      return CP_LABELS.get(int(val), str(val))
    if feature == 'thal':    return THAL_LABELS.get(int(val), str(val))
    if feature == 'slope':   return SLOPE_LABELS.get(int(val), str(val))
    if feature == 'restecg': return RESTECG_LABELS.get(int(val), str(val))
    if feature in ('exang', 'fbs'): return 'Yes' if val else 'No'
    if feature == 'sex':     return 'Male' if val else 'Female'
    return str(val)


def compute_bio_age(d):
    """Rule-based cardiovascular biological age score 0–100. Illustrative only."""
    s = (d['age'] - 20) / 60 * 30
    s += (1 - min(d['thalach'] / max(220 - d['age'], 1), 1)) * 20
    s += d['ca'] / 4 * 15
    s += min(d['oldpeak'] / 6, 1) * 10
    s += min(max(d['trestbps'] - 120, 0) / 80, 1) * 8
    s += min(max(d['chol'] - 180, 0) / 220, 1) * 5
    s += d['exang'] * 4 + d['fbs'] * 4 + (d['slope'] / 2) * 4
    return min(max(int(s), 0), 100)


def bio_color(score):
    return '#22c55e' if score < 35 else '#f59e0b' if score < 60 else '#ef4444'

def bio_label(score):
    return 'Biologically Young' if score < 35 else 'Moderate Aging Signals' if score < 60 else 'Elevated Aging Risk'


# ═══════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    if os.path.exists('heart_model.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('heart_model.pkl'), joblib.load('scaler.pkl'), True
    st.warning('Model files not found — running in demo mode with synthetic data.', icon='⚠️')
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler as SS
    rng = np.random.RandomState(42)
    Xd = rng.randn(400, 13); yd = (Xd[:, 7] < 0).astype(int)
    sc = SS().fit(Xd)
    m  = RandomForestClassifier(100, random_state=42).fit(sc.transform(Xd), yd)
    return m, sc, False

model, scaler, model_loaded = load_artifacts()


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:DM Serif Display,serif;font-size:1.25rem;"
        "color:#f0f2f8;margin-bottom:0.2rem;letter-spacing:-0.01em;'>🫀 Patient Profile</div>"
        "<div style='font-size:0.7rem;color:#2d3250;letter-spacing:0.06em;"
        "margin-bottom:1.2rem;'>Adjust all parameters then click Analyze</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-section'>Demographics</div>", unsafe_allow_html=True)
    age     = st.slider('Age (years)', 20, 80, 54, 1)
    sex     = st.selectbox('Biological Sex', ['Male', 'Female'])
    sex_val = 1 if sex == 'Male' else 0

    st.markdown("<div class='sidebar-section'>Symptoms</div>", unsafe_allow_html=True)
    cp = st.selectbox('Chest Pain Type', [0,1,2,3],
         format_func=lambda x: f'{x} — {CP_LABELS[x]}')
    exang = st.selectbox('Exercise-Induced Angina', [0,1],
            format_func=lambda x: 'Yes' if x else 'No')

    st.markdown("<div class='sidebar-section'>Vitals & Labs</div>", unsafe_allow_html=True)
    trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 130, 1)
    chol     = st.slider('Serum Cholesterol (mg/dl)', 100, 400, 240, 1)
    fbs      = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0,1],
               format_func=lambda x: 'Yes' if x else 'No')
    thalach  = st.slider('Max Heart Rate Achieved (bpm)', 60, 220, 150, 1)

    st.markdown("<div class='sidebar-section'>ECG & Stress Test</div>", unsafe_allow_html=True)
    restecg = st.selectbox('Resting ECG Result', [0,1,2],
              format_func=lambda x: f'{x} — {RESTECG_LABELS[x]}')
    oldpeak = st.slider('ST Depression (oldpeak)', 0.0, 6.0, 1.0, 0.1)
    slope   = st.selectbox('Peak Exercise ST Slope', [0,1,2],
              format_func=lambda x: f'{x} — {SLOPE_LABELS[x]}')

    st.markdown("<div class='sidebar-section'>Imaging & Perfusion</div>", unsafe_allow_html=True)
    ca   = st.slider('Major Vessels via Fluoroscopy (ca)', 0, 4, 0, 1)
    thal = st.selectbox('Thalassemia / Perfusion Type', [0,1,2,3],
           format_func=lambda x: f'{x} — {THAL_LABELS[x]}')

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    predict_btn = st.button('🔬  Analyze Risk Profile', use_container_width=True)


# ═══════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════

st.markdown(
    "<div class='hero-eyebrow'>Longevity Research Tool</div>"
    "<div class='hero-title'>Aging Risk<br>Analyzer</div>"
    "<div class='hero-sub' style='margin-top:0.6rem;'>"
    "An ML-powered cardiovascular risk assessment trained on the Cleveland Heart Disease dataset. "
    "Identifies which biomarkers are aging you fastest — and what to do about it."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

col_main, col_side = st.columns([3, 2], gap='large')

# ── Right column: static info cards ─────────────────────────
with col_side:
    st.markdown("""
    <div class='info-box'>
    <strong style='color:#e2e8f0;'>About this tool</strong><br><br>
    XGBoost model trained on 303 patients and 13 cardiovascular
    biomarkers from the Cleveland Clinic Foundation dataset.<br><br>
    <strong style='color:#9ca3af;'>Core finding:</strong> Functional
    markers — max heart rate, vessel burden, ST changes — outrank
    chronological age as predictors. <em>Biological age beats
    calendar age.</em><br><br>
    <strong style='color:#ef4444;'>⚠ Not a clinical tool.</strong>
    For research and educational use only.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <strong style='color:#e2e8f0;'>Biomarker Quick Reference</strong><br><br>
    🔴 <strong style='color:#fca5a5;'>thalach &lt; 130 bpm</strong>
       — reduced cardiac reserve<br>
    🔴 <strong style='color:#fca5a5;'>ca ≥ 2</strong>
       — significant vessel burden<br>
    🔴 <strong style='color:#fca5a5;'>oldpeak &gt; 2.0</strong>
       — ischemic stress under load<br>
    🟡 <strong style='color:#fde68a;'>slope = 2 (Downsloping)</strong>
       — vascular stiffness<br>
    🟢 <strong style='color:#86efac;'>thalach &gt; 160 bpm</strong>
       — strong cardiac reserve<br>
    🟢 <strong style='color:#86efac;'>ca = 0</strong>
       — clear arteries
    </div>
    """, unsafe_allow_html=True)

# ── Left column: prediction output ──────────────────────────
with col_main:
    if not predict_btn:
        st.markdown("""
        <div style='text-align:center;padding:4rem 1rem;
             background:#13161f;border:1px dashed #1e2235;border-radius:18px;'>
            <div style='font-size:2.8rem;margin-bottom:1rem;'>🫀</div>
            <div style='font-family:DM Serif Display,serif;font-size:1.4rem;
                 color:#2d3250;line-height:1.5;'>
                Configure the patient profile<br>in the sidebar and click Analyze
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Predict ─────────────────────────────────────────
        inputs      = [age, sex_val, cp, trestbps, chol, fbs,
                       restecg, thalach, exang, oldpeak, slope, ca, thal]
        inputs_dict = dict(zip(FEATURE_NAMES, inputs))
        X_sc        = scaler.transform(np.array(inputs).reshape(1, -1))
        prob        = model.predict_proba(X_sc)[0][1]
        pred        = int(prob >= 0.5)
        pct         = prob * 100

        # ── Risk card ────────────────────────────────────────
        if pred == 1:
            cc, ec, vc = 'risk-card risk-card-high', 'risk-eyebrow risk-eyebrow-high', 'risk-verdict risk-verdict-high'
            verdict, icon = 'HIGH RISK', '⚠'
        else:
            cc, ec, vc = 'risk-card risk-card-low', 'risk-eyebrow risk-eyebrow-low', 'risk-verdict risk-verdict-low'
            verdict, icon = 'LOW RISK', '✓'

        st.markdown(
            f"<div class='{cc}'>"
            f"<div class='{ec}'>{icon} &nbsp; Prediction Result</div>"
            f"<div class='{vc}'>{verdict}</div>"
            f"<div class='risk-prob'>Disease probability: "
            f"<strong style='color:#e2e8f0;'>{pct:.1f}%</strong></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Stat pills ───────────────────────────────────────
        st.markdown(
            f"<div class='stat-row'>"
            f"<div class='stat-pill'>Age <span>{age} yrs</span></div>"
            f"<div class='stat-pill'>Sex <span>{sex}</span></div>"
            f"<div class='stat-pill'>Max HR <span>{thalach} bpm</span></div>"
            f"<div class='stat-pill'>Vessels <span>ca = {ca}</span></div>"
            f"<div class='stat-pill'>ST dep. <span>{oldpeak}</span></div>"
            f"<div class='stat-pill'>Cholesterol <span>{chol} mg/dl</span></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Feature contribution chart ───────────────────────
        try:    importances = model.feature_importances_
        except: importances = np.abs(model.coef_[0])

        contrib = pd.Series(np.abs(X_sc[0]) * importances, index=FEATURE_NAMES)
        top5    = contrib.sort_values(ascending=True).tail(5)

        def bar_col(f, r):
            if f == 'thalach': return '#ef4444' if r < 140 else '#22c55e'
            if f in {'ca', 'oldpeak', 'slope', 'thal'}: return '#ef4444' if r > 0 else '#22c55e'
            return '#6b7280'

        colors = [bar_col(f, inputs_dict[f]) for f in top5.index]

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor('#13161f')
        ax.set_facecolor('#13161f')
        bars = ax.barh(
            [FEATURE_LABELS[f] for f in top5.index],
            top5.values, color=colors, edgecolor='none', height=0.50,
        )
        for bar, val in zip(bars, top5.values):
            ax.text(val + max(top5.values) * 0.022,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8, color='#6b7280')
        ax.set_title('Top 5 Risk Contributors — This Profile',
                     fontsize=10, fontweight='bold', color='#9ca3af', pad=10, loc='left')
        ax.set_xlabel('Contribution  (importance × |scaled value|)',
                      fontsize=7.5, color='#4a5070')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(axis='y', colors='#9ca3af', labelsize=8.5)
        ax.tick_params(axis='x', colors='#3d4260', labelsize=7.5)
        ax.legend(
            handles=[mpatches.Patch(color='#ef4444', label='Elevates risk'),
                     mpatches.Patch(color='#22c55e', label='Protective')],
            fontsize=7.5, loc='lower right', framealpha=0, labelcolor='#6b7280',
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ════════════════════════════════════════════════════
        # EXPANDER — What does this mean?
        # ════════════════════════════════════════════════════
        top3      = list(contrib.sort_values(ascending=False).head(3).index)
        bio_score = compute_bio_age(inputs_dict)
        b_color   = bio_color(bio_score)
        b_label   = bio_label(bio_score)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        with st.expander('🧬  What does this mean for me?', expanded=True):

            # ── Top 3 explained ──────────────────────────────
            st.markdown(
                "<div class='section-label'>Your Top 3 Risk Factors — Explained in Plain English</div>",
                unsafe_allow_html=True,
            )
            st.caption('Based on what the model weighted most heavily for your specific biomarker profile.')

            for feat in top3:
                info = PLAIN_ENGLISH.get(feat)
                if not info:
                    continue
                val       = inputs_dict[feat]
                val_label = get_val_label(feat, val)
                is_bad    = (val < info['threshold']) if info['direction'] == 'low_is_bad' \
                            else (val >= info['threshold'])
                signal    = '🔴' if is_bad else '🟢'
                body      = info['bad'] if is_bad else info['good']
                title     = info['title'].format(val=val, val_label=val_label)

                st.markdown(
                    f"<div class='explain-card'>"
                    f"<div class='explain-header'>"
                    f"<span style='font-size:1.3rem;'>{info['icon']}</span>"
                    f"<span style='font-size:1rem;'>{signal}</span>"
                    f"<span class='explain-title'>{title}</span>"
                    f"</div>"
                    f"<div class='explain-body'>{body}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # ── Biology context ───────────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(
                "<div class='section-label'>Why These Biomarkers Matter for Aging</div>",
                unsafe_allow_html=True,
            )
            st.markdown("""
            <div class='explain-card'>
            <div class='explain-body'>
            Modern longevity science has moved beyond counting birthdays.
            Your <strong style='color:#e2e8f0;'>functional biomarkers</strong> — how your heart
            performs under stress, how clear your arteries are, how your body handles glucose — predict
            biological age and long-term mortality far better than your calendar age.<br><br>
            The clearest finding from this dataset:
            <strong style='color:#fca5a5;'>maximum heart rate declines roughly 1 bpm per year of
            biological aging</strong>. If your thalach exceeds what is expected for your age, your
            cardiovascular system is functionally younger than your birth year suggests. The reverse is
            equally true — and more actionable than any birthday.
            </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Action plan ───────────────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(
                "<div class='section-label'>Your Personalised Action Plan</div>",
                unsafe_allow_html=True,
            )

            top_feat  = top3[0]
            val_top   = inputs_dict[top_feat]
            info_top  = PLAIN_ENGLISH.get(top_feat, {})
            is_bad_t  = (val_top < info_top.get('threshold', 1)) \
                        if info_top.get('direction') == 'low_is_bad' \
                        else (val_top >= info_top.get('threshold', 1))
            rec_pool  = RECOMMENDATIONS.get(top_feat, RECOMMENDATIONS['default'])
            recs      = rec_pool.get('bad' if is_bad_t else 'good',
                                     RECOMMENDATIONS['default']['bad'])

            for rec in recs:
                st.markdown(f"<div class='rec-card'>{rec}</div>", unsafe_allow_html=True)

            st.caption(
                'Recommendations are grounded in published cardiology and longevity research. '
                'Always consult a qualified physician before making changes to your health regimen.'
            )

            # ── Biological age score ──────────────────────────
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(
                "<div class='section-label'>Cardiovascular Biological Age Score</div>",
                unsafe_allow_html=True,
            )

            bar_w  = bio_score
            bar_bg = f"<div class='bio-bar-track'><div style='background:{b_color};" \
                     f"width:{bar_w}%;height:8px;border-radius:8px;'></div></div>"

            st.markdown(
                f"<div class='bio-wrap'>"
                f"<div class='bio-num' style='color:{b_color};'>"
                f"{bio_score}"
                f"<span style='font-size:1.1rem;color:#3d4260;'> / 100</span>"
                f"</div>"
                f"<div class='bio-tag' style='color:{b_color};'>{b_label}</div>"
                f"{bar_bg}"
                f"<div style='font-size:0.74rem;color:#3d4260;line-height:1.7;'>"
                f"0 – 35 &nbsp;·&nbsp; Biologically young cardiovascular system<br>"
                f"36 – 60 &nbsp;·&nbsp; Moderate aging signals present<br>"
                f"61 – 100 &nbsp;·&nbsp; Elevated cardiovascular aging risk<br><br>"
                f"<em>Composite of: heart rate reserve, vessel burden, ST markers, "
                f"blood pressure &amp; metabolic indicators.</em>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption('⚠ Biological age score is illustrative and not a validated clinical measurement.')


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════

st.markdown(
    "<div class='footer'>"
    "Built for longevity research &nbsp;·&nbsp; "
    "Cleveland Heart Disease Dataset (UCI / Kaggle) &nbsp;·&nbsp; "
    "XGBoost + Random Forest · scikit-learn &nbsp;·&nbsp; "
    "Not for clinical use"
    "</div>",
    unsafe_allow_html=True,
)
