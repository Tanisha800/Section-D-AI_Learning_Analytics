import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import os

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LearnIQ · AI Study Coach",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&f[]=cabinet-grotesk@300,400,500,700&f[]=jet-brains-mono@400,500&display=swap');

/* ── TOKENS ── */
:root {
    --navy:      #050d1a;
    --navy-2:    #0a1628;
    --navy-3:    #0f2040;
    --navy-4:    #162947;
    --teal:      #00d4b4;
    --teal-dim:  #00a88e;
    --cream:     #f5efe0;
    --cream-2:   #e8dfc8;
    --cream-3:   #c8b99a;
    --muted:     #4a6480;
    --danger:    #ff6b6b;
    --amber:     #ffb347;
    --green:     #4ecb71;
    --border:    rgba(0,212,180,0.12);
    --border-2:  rgba(0,212,180,0.25);
    --glow:      0 0 40px rgba(0,212,180,0.08);
}

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer { visibility: hidden; }

/* keep header functional but invisible */
header {
    background: transparent !important;
}

/* ensure sidebar is visible */
[data-testid="stSidebar"] {
    display: block !important;
}

/* ensure toggle button exists */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed;
    top: 20px;
    left: 10px;
    z-index: 9999;
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Cabinet Grotesk', sans-serif;
    background: var(--navy);
    color: var(--cream);
}
.stApp { background: var(--navy); }
.block-container {
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1140px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--navy-2);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding: 0; }
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] .stRadio > label {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--navy-4); border-radius: 2px; }

/* ── PAGE HERO ── */
.hero {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--navy-2);
    padding: 2.5rem 2.75rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,212,180,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: var(--teal);
}
.hero-title {
    font-family: 'Clash Display', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--cream);
    line-height: 1.05;
    margin-bottom: 0.6rem;
}
.hero-title span { color: var(--teal); }
.hero-sub {
    font-size: 0.95rem;
    color: var(--cream-3);
    font-weight: 400;
    line-height: 1.6;
    max-width: 520px;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── STAT CARDS ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 2rem;
}
.stat-card {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--border-2); }
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--teal), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.stat-card:hover::after { opacity: 1; }
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 500;
    color: var(--cream);
    line-height: 1;
    letter-spacing: -0.03em;
}
.stat-num .unit { font-size: 1rem; color: var(--teal); margin-left: 2px; }
.stat-lbl {
    font-size: 0.69rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 6px;
}
.stat-delta {
    font-size: 0.72rem;
    color: var(--teal);
    margin-top: 4px;
}

/* ── FEATURE CARDS ── */
.feat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; }
.feat-card {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    transition: border-color 0.2s, transform 0.2s;
    cursor: default;
}
.feat-card:hover { border-color: var(--border-2); transform: translateY(-2px); }
.feat-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: rgba(0,212,180,0.08);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    margin-bottom: 1rem;
    color: var(--teal);
    font-weight: 700;
    font-family: 'Clash Display', sans-serif;
}
.feat-title { font-family: 'Clash Display', sans-serif; font-size: 1rem; font-weight: 600; color: var(--cream); margin-bottom: 0.5rem; }
.feat-desc  { font-size: 0.82rem; color: var(--cream-3); line-height: 1.6; }

/* ── BADGE ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 999px;
}
.b-risk { background: rgba(255,107,107,0.12); color: var(--danger); border: 1px solid rgba(255,107,107,0.25); }
.b-avg  { background: rgba(255,179,71,0.12);  color: var(--amber);  border: 1px solid rgba(255,179,71,0.25); }
.b-high { background: rgba(78,203,113,0.12);  color: var(--green);  border: 1px solid rgba(78,203,113,0.25); }
.b-risk::before { content: '▲'; font-size: 0.55rem; }
.b-avg::before  { content: '●'; font-size: 0.55rem; }
.b-high::before { content: '★'; font-size: 0.55rem; }

/* ── RESULT BLOCK ── */
.result-wrap {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-top: 1.5rem;
    box-shadow: var(--glow);
}
.metrics-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1.25rem; }
.metric-tile {
    background: var(--navy-3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.mt-val { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 500; color: var(--cream); letter-spacing: -0.02em; }
.mt-lbl { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px; }
.cat-row { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; }
.cat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.rec-panel {
    background: var(--navy-3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--teal);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
    font-size: 0.87rem;
    color: var(--cream-2);
    line-height: 1.65;
}
.rec-panel b { color: var(--teal); font-weight: 500; }

/* ── INFO/WARN BANNERS ── */
.banner {
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    font-size: 0.82rem;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.b-info { background: rgba(0,212,180,0.06); border: 1px solid rgba(0,212,180,0.2); color: #7de8da; }
.b-warn { background: rgba(255,179,71,0.06); border: 1px solid rgba(255,179,71,0.2); color: #ffc87a; }

/* ── CHAT ── */
.chat-feed { display: flex; flex-direction: column; gap: 10px; margin-bottom: 1.25rem; padding: 0.25rem 0; }
.bubble-wrap-u { display: flex; flex-direction: column; align-items: flex-end; }
.bubble-wrap-a { display: flex; flex-direction: column; align-items: flex-start; }
.bubble-lbl { font-size: 0.62rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
.bubble-u {
    max-width: 74%;
    background: var(--teal);
    color: var(--navy);
    border-radius: 16px 16px 4px 16px;
    padding: 0.7rem 1.1rem;
    font-size: 0.88rem;
    font-weight: 500;
    line-height: 1.55;
}
.bubble-a {
    max-width: 74%;
    background: var(--navy-3);
    color: var(--cream);
    border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 0.7rem 1.1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    white-space: pre-wrap;
}

/* ── FORM ── */
div[data-testid="stForm"] {
    background: var(--navy-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.75rem;
}
.stSelectbox label, .stNumberInput label, .stTextInput label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: var(--navy-3) !important;
    border-color: var(--border) !important;
    color: var(--cream) !important;
    border-radius: 8px !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--teal) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.8rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); padding: 0.55rem 1.1rem;
    border-radius: 0; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--teal) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── SIDEBAR BRAND ── */
.sb-brand {
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.sb-logo {
    font-family: 'Clash Display', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--cream);
    letter-spacing: -0.04em;
}
.sb-logo span { color: var(--teal); }
.sb-tagline { font-size: 0.7rem; color: var(--muted); margin-top: 3px; letter-spacing: 0.06em; }
.sb-status { padding: 1rem 1.5rem; }
.sb-status-title { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem; }
.sb-row { display: flex; align-items: center; gap: 8px; font-size: 0.77rem; color: var(--cream-3); padding: 3px 0; }
.dot-on  { width: 6px; height: 6px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 6px var(--teal); flex-shrink: 0; }
.dot-off { width: 6px; height: 6px; border-radius: 50%; background: var(--amber); flex-shrink: 0; }

/* ── DIVIDER ── */
.fancy-rule {
    display: flex; align-items: center; gap: 12px;
    margin: 2rem 0 1.5rem;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted);
}
.fancy-rule::before, .fancy-rule::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── UPLOAD AREA ── */
[data-testid="stFileUploader"] {
    background: var(--navy-2);
    border: 1px dashed var(--border-2);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        return (
            joblib.load("logistic_model.pkl"),
            joblib.load("linear_model.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("kmeans_model.pkl"),
            joblib.load("cluster_scaler.pkl"),
            joblib.load("feature_columns.pkl"),
        )
    except FileNotFoundError:
        return None, None, None, None, None, None

log_model, lin_model, scaler, kmeans, cluster_scaler, feature_columns = load_artifacts()
models_ok = log_model is not None

@st.cache_resource
def load_agent():
    import traceback
    try:
        from agent.graph import app as ag

        if ag is None:
            raise Exception("Graph compiled but app is None")

        return ag

    except Exception as e:
        error_msg = f"""
❌ AGENT LOAD FAILED

Error:
{str(e)}

Traceback:
{traceback.format_exc()}
"""
        return error_msg  # ⬅️ RETURN ERROR INSTEAD OF None
agent_app = load_agent()
agent_connected = not isinstance(agent_app, str)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def badge_html(cat):
    cls = {"At Risk": "b-risk", "Average": "b-avg", "High Performer": "b-high"}.get(cat, "b-avg")
    return f'<span class="badge {cls}">{cat}</span>'

def run_pred(df_in):
    from app import predict_new_data, generate_recommendations
    r = predict_new_data(df_in, log_model, lin_model, scaler, kmeans, cluster_scaler)
    return generate_recommendations(r)

def setup_mpl():
    rcParams.update({
        "font.family":     "monospace",
        "axes.facecolor":  "#0a1628",
        "figure.facecolor":"#0a1628",
        "text.color":      "#c8b99a",
        "axes.labelcolor": "#4a6480",
        "xtick.color":     "#4a6480",
        "ytick.color":     "#4a6480",
        "axes.edgecolor":  "#0f2040",
        "grid.color":      "#0f2040",
        "grid.linestyle":  "--",
        "grid.alpha":      0.6,
    })

def chart_bar(avg_ds, actual, predicted):
    setup_mpl()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = ["Dataset\nAvg", "Your\nScore", "Profile\nEst."]
    vals   = [avg_ds, actual, predicted]
    colors = ["#162947", "#00d4b4", "#4a6480"]
    bars   = ax.bar(labels, vals, color=colors, width=0.42,
                    edgecolor="#050d1a", linewidth=1.5, zorder=3)
    ax.set_ylim(0, 115)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 2,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color="#f5efe0", fontweight="bold")
    plt.tight_layout(pad=0.5)
    return fig

def chart_cluster(proc_df, actual, predicted, study_hrs):
    setup_mpl()
    proc = proc_df.copy()
    cf   = cluster_scaler.transform(proc[["AverageScore","WklyStudyHours"]])
    proc["Cluster"] = kmeans.predict(cf)
    si   = np.argsort(kmeans.cluster_centers_[:,0])
    lmap = {si[0]:"At Risk", si[1]:"Average", si[2]:"High Performer"}
    proc["Cat"] = proc["Cluster"].map(lmap)
    pal = {"At Risk":"#ff6b6b","Average":"#ffb347","High Performer":"#4ecb71"}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for cat, grp in proc.groupby("Cat"):
        ax.scatter(grp["WklyStudyHours"], grp["AverageScore"],
                   color=pal[cat], alpha=0.18, s=14, label=cat, zorder=2)
    ax.scatter(study_hrs, actual,    color="#00d4b4", s=200,
               edgecolors="#050d1a", linewidths=2, marker="X",
               label="You (current)", zorder=10)
    ax.scatter(study_hrs, predicted, color="#f5efe0", s=110,
               edgecolors="#050d1a", linewidths=1.5, marker="o",
               label="You (profile)", zorder=9)
    ax.set_xlabel("Study hours (encoded)", fontsize=8.5)
    ax.set_ylabel("Average score",         fontsize=8.5)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=8, framealpha=0, labelcolor="#c8b99a",
                    edgecolor="#0f2040")
    plt.tight_layout(pad=0.5)
    return fig

def chart_bulk(results):
    setup_mpl()
    cc = results["Learner Category"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))
    pal_list = ["#ff6b6b","#ffb347","#4ecb71"][:len(cc)]
    wedges, texts, pcts = ax1.pie(
        cc.values, labels=cc.index, colors=pal_list,
        autopct="%1.0f%%", startangle=140,
        textprops={"fontsize":8,"color":"#c8b99a"},
        pctdistance=0.78,
        wedgeprops={"linewidth":2,"edgecolor":"#050d1a"}
    )
    for pct in pcts: pct.set_color("#f5efe0"); pct.set_fontweight("bold")
    ax1.set_title("Learner categories", fontsize=8.5, color="#4a6480", pad=8, loc="left")
    ax2.hist(results["AverageScore"], bins=20, color="#00d4b4",
             edgecolor="#050d1a", alpha=0.7, linewidth=0.8, zorder=3)
    ax2.set_xlabel("Score",  fontsize=8.5)
    ax2.set_ylabel("Count",  fontsize=8.5)
    ax2.set_title("Score distribution", fontsize=8.5, color="#4a6480", pad=8, loc="left")
    ax2.spines[:].set_visible(False)
    ax2.yaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    plt.tight_layout(pad=0.6)
    return fig


st.markdown("""
<style>
[data-testid="stSidebar"] > div:first-child {
    padding-left: 20px;  /* Adjust value as needed */
}
</style>
""", unsafe_allow_html=True)
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar :
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-logo">Learn<span>IQ</span></div>
        <div class="sb-tagline">AI Learning Analytics · Milestone 2</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Dashboard", "Individual", "Bulk Analysis", "AI Coach"],
        label_visibility="collapsed"
    )

    st.markdown("""<div class="sb-status">
        <div class="sb-status-title">System status</div>
    """, unsafe_allow_html=True)

    for name, ok in [
        ("Logistic regression", models_ok),
        ("Linear regression",   models_ok),
        ("K-means clustering",  models_ok),
        ("LangGraph agent",     agent_app is not None),
    ]:
        dot = "dot-on" if ok else "dot-off"
        st.markdown(f'<div class="sb-row"><div class="{dot}"></div>{name}</div>',
                    unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">Project 2 · Section D</div>
        <div class="hero-title">Predict. Analyse.<br><span>Coach.</span></div>
        <div class="hero-sub">
            An AI-powered learning analytics system that predicts student performance,
            identifies learning gaps, and delivers personalised coaching through an
            agentic study coach.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Model performance</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-num">73<span class="unit">%</span></div>
            <div class="stat-lbl">Classification accuracy</div>
            <div class="stat-delta">↑ Logistic regression</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">0.85</div>
            <div class="stat-lbl">F1 score</div>
            <div class="stat-delta">↑ High recall priority</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">16.2</div>
            <div class="stat-lbl">Regression RMSE</div>
            <div class="stat-delta">Linear regression</div>
        </div>
        <div class="stat-card">
            <div class="stat-num">30<span class="unit">K+</span></div>
            <div class="stat-lbl">Training records</div>
            <div class="stat-delta">Kaggle dataset</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-rule">Features</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">01</div>
            <div class="feat-title">Individual Analysis</div>
            <div class="feat-desc">Enter a student's profile and scores. Get an instant pass/fail prediction, learner category classification, and a personalised study recommendation.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">02</div>
            <div class="feat-title">Bulk Analysis</div>
            <div class="feat-desc">Upload a CSV to analyse your entire cohort. Get category breakdowns, score distributions, and download a results file with predictions for every student.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">03</div>
            <div class="feat-title">AI Study Coach</div>
            <div class="feat-desc">Chat with a LangGraph-powered agentic coach. Analyse performance data, generate weekly study plans, and surface targeted learning resources.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-rule">Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-card" style="margin-top:0">
        <div class="feat-desc" style="line-height:1.8">
            <span style="color:#f5efe0;font-weight:700">Students Performance in Exams</span>
            &nbsp;·&nbsp; Kaggle &nbsp;·&nbsp; 30,000+ records<br>
            Features: Math · Reading · Writing scores &nbsp;·&nbsp;
            Ethnic group · Parent education · Lunch type ·
            Test preparation · Marital status · Sport practice ·
            Siblings · Transport · Weekly study hours
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── INDIVIDUAL ────────────────────────────────────────────────────────────────
elif page == "Individual":
    st.markdown("""
    <div class="hero-eyebrow" style="margin-bottom:0.4rem">Analysis</div>
    <div class="hero-title" style="font-size:1.9rem;margin-bottom:0.4rem">Individual Prediction</div>
    <div class="hero-sub" style="margin-bottom:1.75rem">Fill in the student profile below to get a real-time prediction.</div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.markdown('<div class="banner b-warn">Models not loaded — run <code>app.py</code> first to generate .pkl files.</div>',
                    unsafe_allow_html=True)
        st.stop()

    with st.form("ind_form"):
        st.markdown('<div class="sec-label">Student background</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender         = st.selectbox("Gender",               ["female","male"])
            ethnic_group   = st.selectbox("Ethnic group",         ["group A","group B","group C","group D","group E"])
            parent_educ    = st.selectbox("Parent education",     ["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
            lunch_type     = st.selectbox("Lunch type",            ["standard","free/reduced"])
        with c2:
            test_prep      = st.selectbox("Test preparation",     ["none","completed"])
            parent_marital = st.selectbox("Parent marital status",["married","single","widowed","divorced"])
            practice_sport = st.selectbox("Sport practice",       ["regularly","sometimes","never"])
            is_first_child = st.selectbox("First child",          ["yes","no"])
        with c3:
            nr_siblings    = st.number_input("Siblings",           0, 10, 1)
            transport      = st.selectbox("Transport",             ["school_bus","private"])
            wkly_hours     = st.selectbox("Weekly study hours",    ["< 5","5 - 10","> 10"])

        st.markdown('<div class="sec-label" style="margin-top:1.25rem">Current scores</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1: math_score    = st.number_input("Math score",    0, 100, 70)
        with s2: reading_score = st.number_input("Reading score", 0, 100, 70)
        with s3: writing_score = st.number_input("Writing score", 0, 100, 70)

        go = st.form_submit_button("⬡  Run Analysis", use_container_width=True)

    if go:
        df_in = pd.DataFrame([{
            "Gender": gender, "EthnicGroup": ethnic_group, "ParentEduc": parent_educ,
            "LunchType": lunch_type, "TestPrep": test_prep,
            "ParentMaritalStatus": parent_marital, "PracticeSport": practice_sport,
            "IsFirstChild": is_first_child, "NrSiblings": nr_siblings,
            "TransportMeans": transport, "WklyStudyHours": wkly_hours,
            "MathScore": math_score, "ReadingScore": reading_score, "WritingScore": writing_score
        }])

        with st.spinner("Running models…"):
            try:
                res          = run_pred(df_in)
                pred_pf      = res["Predicted_PassFail"][0]
                pred_score   = res["Predicted_AverageScore"][0]
                actual_score = res["AverageScore"][0]
                cat          = res["Learner Category"][0]
                rec          = res["Recommendation"][0]
                study_hrs    = res["WklyStudyHours"][0]

                st.markdown("""
                <div class="banner b-info">
                    <b>How to read this:</b> "Profile estimate" is based on your
                    demographic & study habits. "Your score" is from the scores you entered above.
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="result-wrap">
                    <div class="metrics-row">
                        <div class="metric-tile">
                            <div class="mt-val">{actual_score:.1f}</div>
                            <div class="mt-lbl">Actual avg score</div>
                        </div>
                        <div class="metric-tile">
                            <div class="mt-val" style="color:{'#4ecb71' if actual_score>=40 else '#ff6b6b'}">
                                {"Pass" if actual_score>=40 else "Fail"}
                            </div>
                            <div class="mt-lbl">Actual status</div>
                        </div>
                        <div class="metric-tile">
                            <div class="mt-val" style="color:#00d4b4">{pred_score:.1f}</div>
                            <div class="mt-lbl">Profile estimate</div>
                        </div>
                        <div class="metric-tile">
                            <div class="mt-val" style="color:{'#4ecb71' if pred_pf==1 else '#ff6b6b'}">
                                {"Pass" if pred_pf==1 else "Fail"}
                            </div>
                            <div class="mt-lbl">Profile status</div>
                        </div>
                    </div>
                    <div class="cat-row">
                        <span class="cat-label">Learner category</span>
                        {badge_html(cat)}
                    </div>
                    <div class="rec-panel">
                        <b>Recommendation</b><br>{rec}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="fancy-rule" style="margin-top:2rem">Visualisations</div>',
                            unsafe_allow_html=True)
                tab1, tab2 = st.tabs(["Score Comparison", "Cluster Map"])

                with tab1:
                    try:
                        from app import load_data, preprocess_data
                        proc = preprocess_data(load_data("./Data/raw/Student_Performance.csv"))
                        st.pyplot(chart_bar(proc["AverageScore"].mean(), actual_score, pred_score),
                                  use_container_width=True)
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")

                with tab2:
                    try:
                        from app import load_data, preprocess_data
                        proc = preprocess_data(load_data("./Data/raw/Student_Performance.csv"))
                        st.pyplot(chart_cluster(proc, actual_score, pred_score, study_hrs),
                                  use_container_width=True)
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ── BULK ANALYSIS ─────────────────────────────────────────────────────────────
elif page == "Bulk Analysis":
    st.markdown("""
    <div class="hero-eyebrow" style="margin-bottom:0.4rem">Analysis</div>
    <div class="hero-title" style="font-size:1.9rem;margin-bottom:0.4rem">Bulk Cohort Analysis</div>
    <div class="hero-sub" style="margin-bottom:1.75rem">Upload a student CSV to analyse an entire cohort at once.</div>
    """, unsafe_allow_html=True)

    if not models_ok:
        st.markdown('<div class="banner b-warn">Models not loaded — run <code>app.py</code> first.</div>',
                    unsafe_allow_html=True)
        st.stop()

    st.markdown("""
    <div class="banner b-info">
        Required columns: <code>Gender · EthnicGroup · ParentEduc · LunchType · TestPrep ·
        ParentMaritalStatus · PracticeSport · IsFirstChild · NrSiblings · TransportMeans ·
        WklyStudyHours · MathScore · ReadingScore · WritingScore</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"],
                                 label_visibility="collapsed")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.caption(f"{len(df_up)} students loaded")
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("⬡  Run Batch Predictions", use_container_width=True):
            with st.spinner("Analysing cohort…"):
                try:
                    results = run_pred(df_up.copy()).drop(columns=["Cluster"], errors="ignore")
                    cc = results["Learner Category"].value_counts()

                    st.markdown(f"""
                    <div class="stat-grid" style="margin-top:1.5rem">
                        <div class="stat-card">
                            <div class="stat-num">{results["AverageScore"].mean():.1f}</div>
                            <div class="stat-lbl">Cohort mean score</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num" style="color:#ff6b6b">{cc.get("At Risk",0)}</div>
                            <div class="stat-lbl">At risk</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num" style="color:#ffb347">{cc.get("Average",0)}</div>
                            <div class="stat-lbl">Average</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-num" style="color:#4ecb71">{cc.get("High Performer",0)}</div>
                            <div class="stat-lbl">High performers</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    tab_t, tab_c = st.tabs(["Results Table", "Charts"])
                    with tab_t:
                        st.dataframe(results, use_container_width=True)
                        st.download_button(
                            "⬡  Download CSV",
                            results.to_csv(index=False).encode(),
                            "predictions.csv", "text/csv"
                        )
                    with tab_c:
                        st.pyplot(chart_bulk(results), use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")


# ── AI COACH ──────────────────────────────────────────────────────────────────
elif page == "AI Coach":
    st.markdown("""
    <div class="hero-eyebrow" style="margin-bottom:0.4rem">Milestone 2</div>
    <div class="hero-title" style="font-size:1.9rem;margin-bottom:0.4rem">AI Study Coach</div>
    <div class="hero-sub" style="margin-bottom:1.75rem">
        Chat with the LangGraph-powered agent — analyse data, build study plans, find resources.
    </div>
    """, unsafe_allow_html=True)

    if not agent_connected:
        st.markdown("""
        <div class="banner b-warn">
            Agent not connected. Ensure <code>agent/graph.py</code>, <code>agent/llm.py</code>,
            <code>agent/tools.py</code>, and <code>agent/rag.py</code> are present and installed.
            Running in demo mode.
        </div>
        """, unsafe_allow_html=True)

    
    agent_csv = st.file_uploader("Attach Student CSV", type=["csv"], key="acsv")
    if agent_csv:
        tmp = "/tmp/agent_upload.csv"
        with open(tmp, "wb") as f: f.write(agent_csv.getbuffer())
        st.session_state["acsv_path"] = tmp
        st.caption(f"Attached: {agent_csv.name}")

    st.markdown("""
    <div class="banner b-info">
        Try: &nbsp;<b>"analyze student"</b>&nbsp; · &nbsp;<b>"create a study plan"</b>&nbsp; · &nbsp;<b>"find resources for math"</b>
    </div>
    """, unsafe_allow_html=True)

    if "chat"   not in st.session_state: st.session_state.chat   = []
    if "astate" not in st.session_state: st.session_state.astate = {}

    # Chat history
    if st.session_state.chat:
        html = '<div class="chat-feed">'
        for role, msg in st.session_state.chat:
            if role == "user":
                html += f'<div class="bubble-wrap-u"><div class="bubble-lbl">You</div><div class="bubble-u">{msg}</div></div>'
            else:
                html += f'<div class="bubble-wrap-a"><div class="bubble-lbl">Coach</div><div class="bubble-a">{msg}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_msg = st.text_input(
            "Message",
            placeholder="e.g. create a study plan for an at-risk student",
            label_visibility="collapsed"
        )
        col_s, col_c = st.columns([5,1])
        with col_s: send  = st.form_submit_button("⬡  Send",  use_container_width=True)
        with col_c: clear = st.form_submit_button("Clear",    use_container_width=True)

    if clear:
        st.session_state.chat   = []
        st.session_state.astate = {}
        st.rerun()

    if send and user_msg.strip():
        st.session_state.chat.append(("user", user_msg))
        with st.spinner("Coach is thinking…"):
            try:
                if agent_connected:
                    state = dict(st.session_state.astate)
                    state["input"] = user_msg
                    if "acsv_path" in st.session_state:
                        state["file"] = st.session_state["acsv_path"]
                    result = agent_app.invoke(state)
                    st.session_state.astate.update(result)
                    if "analysis"   in result: reply = "Analysis summary:\n\n"  + str(result["analysis"])
                    elif "plan"     in result: reply = "Your study plan:\n\n"   + str(result["plan"])
                    elif "resources" in result:
                        reply = "Learning resources:\n\n" + str(result["resources"])
                        if result.get("links"):
                            reply += "\n\nLinks:\n" + "\n".join(f"• {l}" for l in result["links"])
                    elif "response" in result: reply = result["response"]
                    else: reply = "Could not generate a response — please rephrase."
                else:
                    u = user_msg.lower()
                    if "plan" in u:
                        reply = ("Here's a sample study plan:\n\n"
                                 "1. Review weak subjects — 30 min daily\n"
                                 "2. Full practice test every weekend\n"
                                 "3. Focus on fundamentals if score < 60\n"
                                 "4. Active recall only — no passive re-reading\n"
                                 "5. Prioritise 7–8 hrs sleep; cramming is counterproductive\n\n"
                                 "→ Connect the agent for a fully personalised plan.")
                    elif "resource" in u or "material" in u:
                        reply = ("Recommended resources:\n\n"
                                 "• Khan Academy — khanacademy.org\n"
                                 "• MIT OpenCourseWare — ocw.mit.edu\n"
                                 "• 3Blue1Brown (YouTube) — visual maths\n"
                                 "• Crash Course (YouTube) — all subjects\n\n"
                                 "→ Connect the agent for topic-specific links.")
                    elif "analyz" in u:
                        reply = "Upload a CSV in the panel above and ensure the agent is connected to run a full performance analysis."
                    else:
                        reply = ("Hi — I'm your AI Study Coach.\n\n"
                                 "I can help you:\n"
                                 "• Analyse student performance data\n"
                                 "• Build a personalised weekly study plan\n"
                                 "• Find curated learning resources\n\n"
                                 "Try: 'create a study plan' or 'find resources for reading'.")
            except Exception as e:
                reply = f"Agent error: {e}"

        st.session_state.chat.append(("ai", reply))
        st.rerun()