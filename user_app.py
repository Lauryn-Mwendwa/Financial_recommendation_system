import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib, os, pickle

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FinSight — Your Financial Advisor",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS — ShieldPay Inspired Light Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.stApp {
    background: #F0F4FF;
    color: #1A1F36;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Top Nav Bar ── */
.navbar {
    background: #FFFFFF;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 12px rgba(26,31,54,0.06);
    position: sticky;
    top: 0;
    z-index: 999;
    margin-bottom: 0;
}
.nav-logo {
    font-size: 1.4rem;
    font-weight: 800;
    color: #1A1F36;
    letter-spacing: -0.03em;
}
.nav-logo span { color: #4361EE; }
.nav-pill {
    background: #EEF2FF;
    color: #4361EE;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* ── Page Wrapper ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
    display: flex;
    justify-content: center;
}

.navbar {
    width: 100%;
    max-width: 780px;
    margin: 0 auto;
}

.page-wrap {
    width: 100%;
    max-width: 780px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
    box-sizing: border-box;
}

/* ── Step Header ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #EEF2FF;
    color: #4361EE;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.page-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #1A1F36;
    line-height: 1.2;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.page-sub {
    color: #6B7280;
    font-size: 0.92rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* ── Cards ── */
.fin-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 20px rgba(26,31,54,0.06);
    border: 1px solid rgba(26,31,54,0.05);
}
.fin-card-blue {
    background: linear-gradient(135deg, #4361EE 0%, #3A56D4 100%);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 30px rgba(67,97,238,0.3);
    color: white;
}
.fin-card-yellow {
    background: linear-gradient(135deg, #F7B731 0%, #F59E0B 100%);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 30px rgba(247,183,49,0.3);
    color: #1A1F36;
}
.fin-card-dark {
    background: linear-gradient(135deg, #1A1F36 0%, #2D3561 100%);
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 30px rgba(26,31,54,0.2);
    color: white;
}

/* ── Card Title ── */
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9CA3AF;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.card-title-white {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.6);
    margin-bottom: 1rem;
}

/* ── Input Labels ── */
.stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    margin-bottom: 0.3rem !important;
}

/* ── Input Fields ── */
.stNumberInput input, .stTextInput input {
    background: #F9FAFB !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 12px !important;
    color: #1A1F36 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #4361EE !important;
    box-shadow: 0 0 0 3px rgba(67,97,238,0.1) !important;
    background: #FFFFFF !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #F9FAFB !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 12px !important;
    color: #1A1F36 !important;
    font-size: 0.92rem !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: #4361EE !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4361EE 0%, #3A56D4 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(67,97,238,0.3) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(67,97,238,0.4) !important;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.metric-item {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1A1F36;
    letter-spacing: -0.02em;
}

/* ── Persona Banner ── */
.persona-banner {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.persona-icon {
    font-size: 2.2rem;
    flex-shrink: 0;
}
.persona-name {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.persona-desc {
    font-size: 0.82rem;
    margin-top: 0.2rem;
    opacity: 0.8;
}

/* ── Rec Cards ── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    line-height: 1.6;
    font-weight: 500;
}
.rec-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: 0.1rem; }
.rec-success { background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; }
.rec-warn    { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.rec-danger  { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
.rec-info    { background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE; }

/* ── Progress Bar ── */
.prog-wrap {
    background: #F3F4F6;
    border-radius: 50px;
    height: 10px;
    margin: 0.6rem 0;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: #F3F4F6;
    margin: 1.5rem 0;
}

/* ── Step Indicator ── */
.steps-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-bottom: 2rem;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #E5E7EB;
}
.step-dot-active {
    width: 24px; height: 8px;
    border-radius: 50px;
    background: #4361EE;
}
.step-line {
    width: 30px; height: 2px;
    background: #E5E7EB;
    border-radius: 2px;
}

/* ── Goal Info Row ── */
.goal-row {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #F3F4F6;
    font-size: 0.88rem;
}
.goal-row:last-child { border-bottom: none; }
.gl { color: #6B7280; font-weight: 500; }
.gv { color: #1A1F36; font-weight: 700; }

/* ── Status Pill ── */
.status-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 700;
}
.pill-green  { background: #D1FAE5; color: #065F46; }
.pill-yellow { background: #FEF3C7; color: #92400E; }
.pill-red    { background: #FEE2E2; color: #991B1B; }

hr { border-color: #F3F4F6 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# TRAIN MODEL ON CSV DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    import joblib
    scaler = joblib.load("C:/Users/Administrator/Documents/My Documents/my_school_project/models/scaler.pkl")
    km     = joblib.load("C:/Users/Administrator/Documents/My Documents/my_school_project/models/kmeans_model.pkl")
    return scaler, km

scaler, km = load_model()


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def predict_persona(savings_rate, debt_to_income, spending_ratio,
                    discretionary_ratio, subscription_ratio):
    X = scaler.transform([[savings_rate, debt_to_income, spending_ratio,
                           discretionary_ratio, subscription_ratio]])
    cluster = km.predict(X)[0]
    persona_map = {
        0: ("The Controlled Spender", "💎", "#4361EE", "#EEF2FF",
            "You manage your money well with controlled spending habits."),
        1: ("The Stretched Saver",    "⚡", "#F7B731", "#FFFBEB",
            "You save well but your expenses are pushing your limits."),
        2: ("The Debt Burdened",      "⚠️", "#F59E0B", "#FFF7ED",
            "High debt levels are limiting your financial progress."),
        3: ("The High Risk Spender",  "🔴", "#EF4444", "#FEF2F2",
            "Your spending significantly exceeds your income — action needed.")
    }
    return persona_map[cluster]


def classify_feasibility(surplus, required):
    if surplus <= 0:
        return "Not Feasible", "pill-red", 0
    ratio = surplus / required if required > 0 else 1
    if ratio >= 1.0:   return "Feasible",           "pill-green",  min(ratio, 1.0)
    elif ratio >= 0.6: return "Partially Feasible",  "pill-yellow", ratio
    else:              return "Not Feasible",         "pill-red",    ratio


def get_recommendations(persona_name, feasibility, savings_rate,
                         spending_ratio, discretionary_ratio, subscription_ratio):
    recs = []
    if persona_name == "The Controlled Spender":
        recs.append(("success", "✅", "Your spending is well controlled — excellent financial discipline!"))
        if savings_rate < 0.20:
            recs.append(("info", "💰", "Try increasing your savings rate to at least 20% of your income."))
        else:
            recs.append(("success", "📈", "Consider moving surplus savings into a money market or investment account."))

    elif persona_name == "The Stretched Saver":
        recs.append(("warn", "⚠️", "You save well but your total spending exceeds your income — unsustainable long term."))
        recs.append(("warn", "🔍", "Review your largest expense categories and identify areas to cut by 10–15%."))
        if discretionary_ratio > 0.15:
            recs.append(("warn", "🎮", "Your discretionary spending is high — set a strict monthly entertainment budget."))

    elif persona_name == "The Debt Burdened":
        recs.append(("danger", "🚨", "Your debt-to-income ratio is critically high — make debt repayment your top priority."))
        recs.append(("danger", "💳", "Focus on paying off high-interest debt first using the avalanche method."))
        recs.append(("warn",   "🚫", "Avoid taking on any new debt until your ratio drops below 0.4."))
        if savings_rate < 0.10:
            recs.append(("info", "💰", "Even saving 5–10% consistently will build a buffer against further debt."))

    elif persona_name == "The High Risk Spender":
        recs.append(("danger", "🔴", "Your finances are in a critical state — immediate corrective action is required."))
        recs.append(("danger", "📉", "Your spending is nearly double your income — create a strict budget immediately."))
        if subscription_ratio > 0.10:
            recs.append(("warn", "📱", "Very high subscription costs — audit and cancel all non-essential subscriptions."))
        recs.append(("danger", "🏦", "Speak with a financial advisor about debt restructuring options."))

    if feasibility == "Feasible":
        recs.append(("success", "🏆", "Great news — your financial goal is achievable within your set timeline!"))
        if savings_rate > 0.50:
            recs.append(("success", "🌟", "Excellent savings rate! You may achieve your goal ahead of schedule."))
    elif feasibility == "Partially Feasible":
        recs.append(("warn", "⏳", "Your goal is partially achievable — you are about 60–99% of the way there."))
        recs.append(("warn", "📊", "Reducing monthly expenses by just 10% could make your goal fully feasible."))
        recs.append(("info", "📅", "Consider extending your goal timeline by 3–6 months as an alternative."))
    elif feasibility == "Not Feasible":
        recs.append(("danger", "❌", "Your goal is not achievable at your current savings pace."))
        recs.append(("warn",   "🔄", "Options: increase income, reduce expenses, lower your target, or extend timeline."))
        if spending_ratio > 1.0:
            recs.append(("danger", "⚡", "Critical: You are spending more than you earn — address this before pursuing any goal."))

    return recs


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"  # "login" or "signup"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "profile_pic" not in st.session_state:
    st.session_state.profile_pic = None

# Apply the selected theme on every rerun
if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp, .main, .block-container, .page-wrap { background: #0F172A !important; color: #E2E8F0 !important; }
    .navbar { background: #111827 !important; box-shadow: 0 1px 18px rgba(0,0,0,0.35) !important; }
    .nav-pill { background: #1F2937 !important; color: #C7D2FE !important; }
    .fin-card, .fin-card-blue, .fin-card-yellow, .fin-card-dark { background: #111827 !important; border-color: rgba(148,163,184,0.18) !important; color: #E5E7EB !important; }
    .card-title, .card-title-white, .page-title, .page-sub, .step-badge, .metric-label, .metric-value, .goal-row, .gl, .gv { color: #E5E7EB !important; }
    .step-badge { background: #1F2937 !important; color: #C7D2FE !important; }
    .stButton>button { background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div { background: #111827 !important; border-color: #334155 !important; color: #E5E7EB !important; }
    .stSlider > div > div > div > div { background: #3B82F6 !important; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# USER DATABASE MANAGEMENT
# ─────────────────────────────────────────
import json
import os

USER_DATA_FILE = "user_data.json"

def load_user_data():
    """Load user data from JSON file"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(data):
    """Save user data to JSON file"""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def get_user_data():
    """Get user data, loading from file if needed"""
    if not hasattr(st.session_state, 'user_database'):
        st.session_state.user_database = load_user_data()
    return st.session_state.user_database

def add_user(email, password, name):
    """Add a new user to the database"""
    users = get_user_data()
    users[email] = {
        "password": password,  # In production, hash this!
        "name": name,
        "dob": "",
        "profile_pic": None
    }
    save_user_data(users)
    st.session_state.user_database = users

def user_exists(email):
    """Check if user exists"""
    users = get_user_data()
    return email in users

def validate_login(email, password):
    """Validate login credentials"""
    users = get_user_data()
    if email in users and users[email]["password"] == password:
        return users[email]["name"]
    return None


# ─────────────────────────────────────────
# AUTHENTICATION FUNCTIONS
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# AUTHENTICATION FUNCTIONS
# ─────────────────────────────────────────
def show_auth():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # Auth Header
    if st.session_state.auth_mode == "signup":
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:0.8rem">✨</div>
            <h1 style="font-size:2.2rem;font-weight:800;color:#1A1F36;letter-spacing:-0.03em;margin:0">
                Create Your<br>FinSight Account
            </h1>
            <p style="color:#6B7280;font-size:1rem;margin-top:0.8rem;font-weight:400;max-width:400px;margin-left:auto;margin-right:auto">
                Join thousands of users who trust FinSight with their financial journey. Your data stays secure and private.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:0.8rem">🔐</div>
            <h1 style="font-size:2.2rem;font-weight:800;color:#1A1F36;letter-spacing:-0.03em;margin:0">
                Welcome Back<br>to FinSight
            </h1>
            <p style="color:#6B7280;font-size:1rem;margin-top:0.8rem;font-weight:400;max-width:400px;margin-left:auto;margin-right:auto">
                Your financial data is protected. Please login with your credentials to access your personalized financial advisor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Auth Form
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)

    if st.session_state.auth_mode == "signup":
        st.markdown('<div class="card-title">📝 Create Account</div>', unsafe_allow_html=True)

        name = st.text_input("Full Name", key="signup_name", placeholder="Enter your full name")
        email = st.text_input("Email Address", key="signup_email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm", placeholder="Confirm your password")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Create Account", key="signup_btn", use_container_width=True):
                if not all([name, email, password, confirm_password]):
                    st.error("❌ Please fill in all fields.")
                elif password != confirm_password:
                    st.error("❌ Passwords don't match.")
                elif user_exists(email):
                    st.error("❌ An account with this email already exists.")
                    if st.button("🔄 Switch to Login", key="switch_to_login"):
                        st.session_state.auth_mode = "login"
                        st.rerun()
                elif len(password) < 6:
                    st.error("❌ Password must be at least 6 characters long.")
                else:
                    add_user(email, password, name)
                    st.session_state.logged_in = True
                    st.session_state.user_name = name
                    st.session_state.user_email = email
                    st.success(f"🎉 Welcome to FinSight, {name}!")
                    st.rerun()

    else:  # Login mode
        st.markdown('<div class="card-title">🔑 Login Credentials</div>', unsafe_allow_html=True)

        email = st.text_input("Email Address", key="login_email", placeholder="your@email.com")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔓 Login", key="login_btn", use_container_width=True):
                if email and password:
                    user_name = validate_login(email, password)
                    if user_name:
                        st.session_state.logged_in = True
                        st.session_state.user_name = user_name
                        st.session_state.user_email = email
                        st.success(f"Welcome back, {user_name}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both email and password.")

    # Mode Switcher
    st.markdown('<div style="text-align:center;margin-top:1.5rem;padding-top:1rem;border-top:1px solid #E5E7EB">', unsafe_allow_html=True)

    if st.session_state.auth_mode == "signup":
        st.markdown("""
        <p style="color:#9CA3AF;font-size:0.85rem;margin:0">
            Already have an account?
        </p>
        """, unsafe_allow_html=True)
        if st.button("🔄 Switch to Login", key="switch_to_login_main"):
            st.session_state.auth_mode = "login"
            st.rerun()
    else:
        st.markdown("""
        <p style="color:#9CA3AF;font-size:0.85rem;margin:0">
            New to FinSight?
        </p>
        """, unsafe_allow_html=True)
        if st.button("✨ Create Account", key="switch_to_signup_main"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    st.markdown("""
    <p style="color:#9CA3AF;font-size:0.8rem;margin:0.5rem 0 0 0">
        🔒 Your data stays secure and private
    </p>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_account():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">Account · Personal Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Profile & Settings</div>', unsafe_allow_html=True)

    users = get_user_data()
    user = users.get(st.session_state.user_email, {})
    name = st.text_input("Full Name", value=user.get("name", ""), key="account_name")
    dob = st.text_input("Date of Birth", value=user.get("dob", ""), key="account_dob")
    email = st.text_input("Email", value=st.session_state.user_email, disabled=True)

    profile_file = st.file_uploader("Upload Profile Picture", type=["png", "jpg", "jpeg"], key="account_profile_pic")
    if profile_file is not None:
        st.session_state.profile_pic = profile_file.read()
    if st.session_state.profile_pic:
        st.image(st.session_state.profile_pic, width=140)

    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔏 Account Credentials</div>', unsafe_allow_html=True)
    current_password = st.text_input("Current Password", type="password", key="account_current_password")
    new_password = st.text_input("New Password", type="password", key="account_new_password")
    confirm_password = st.text_input("Confirm New Password", type="password", key="account_confirm_password")

    if st.button("Save Profile", key="save_profile"):
        errors = []
        if not name:
            errors.append("Please enter your full name.")
        if new_password and new_password != confirm_password:
            errors.append("New passwords must match.")
        if new_password and not current_password:
            errors.append("Enter your current password to update your password.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            if new_password:
                if validate_login(st.session_state.user_email, current_password) is None:
                    st.error("❌ Current password is incorrect.")
                    return
                users[st.session_state.user_email]["password"] = new_password
            users[st.session_state.user_email]["name"] = name
            users[st.session_state.user_email]["dob"] = dob
            save_user_data(users)
            st.session_state.user_name = name
            st.success("✅ Profile updated successfully.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_settings():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="step-badge">App Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Appearance & Preferences</div>', unsafe_allow_html=True)

    theme_choice = st.radio("Theme Mode", ["Light", "Dark"], index=1 if st.session_state.dark_mode else 0)
    st.session_state.dark_mode = theme_choice == "Dark"

    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎨 Display Settings</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='color:#6B7280;font-size:0.92rem'>Current theme: <strong>{theme_choice}</strong></p>", unsafe_allow_html=True)
    st.markdown('<div style="margin-top:1rem;color:#6B7280;font-size:0.92rem">Toggle between light and dark mode for a more comfortable reading experience.</div>', unsafe_allow_html=True)
    if st.button("Save Settings", key="save_settings"):
        st.success("✅ Settings saved.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown(f"### {st.session_state.user_name}")
    st.sidebar.markdown(f"📧 {st.session_state.user_email}")
    if st.session_state.profile_pic:
        st.sidebar.image(st.session_state.profile_pic, width=140)
    st.sidebar.markdown("---")
    nav_items = ["Dashboard", "Goals", "Account", "Settings"]
    current = "Dashboard"
    if st.session_state.page == "goal":
        current = "Goals"
    elif st.session_state.page == "account":
        current = "Account"
    elif st.session_state.page == "settings":
        current = "Settings"
    selection = st.sidebar.radio("Navigation", nav_items, index=nav_items.index(current))
    page_map = {
        "Dashboard": "home",
        "Goals": "goal",
        "Account": "account",
        "Settings": "settings"
    }
    st.session_state.page = page_map[selection]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick Actions**")
    if st.sidebar.button("🚀 Start Analysis", key="sidebar_start"):
        st.session_state.page = "income"
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Theme**")
    theme_choice = st.sidebar.radio("", ["Light", "Dark"], index=1 if st.session_state.dark_mode else 0, key="sidebar_theme")
    st.session_state.dark_mode = theme_choice == "Dark"
    st.sidebar.markdown("---")
    show_logout()


def show_logout():
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.user_email = ""
        st.session_state.page = "home"
        st.rerun()


# ─────────────────────────────────────────
# NAV BAR
# ─────────────────────────────────────────
if st.session_state.logged_in:
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-logo">Fin<span>Sight</span></div>
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="color:#6B7280; font-size:0.9rem; font-weight:500;">Hey {st.session_state.user_name} 👋</div>
            <div class="nav-pill">💎 Smart Finance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="navbar">
        <div class="nav-logo">Fin<span>Sight</span></div>
        <div class="nav-pill">💎 Smart Finance</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────
def show_home():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    # Hero
    if st.session_state.logged_in:
        st.markdown(f"""
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:0.8rem">👋</div>
            <h1 style="font-size:2.2rem;font-weight:800;color:#1A1F36;letter-spacing:-0.03em;margin:0">
                Hey {st.session_state.user_name}!<br>Welcome back
            </h1>
            <p style="color:#6B7280;font-size:1rem;margin-top:0.8rem;font-weight:400;max-width:400px;margin-left:auto;margin-right:auto">
                Ready to analyze your finances? Let's get personalized insights and recommendations for your financial goals.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
            <div style="font-size:3.5rem; margin-bottom:0.8rem">💎</div>
            <h1 style="font-size:2.2rem;font-weight:800;color:#1A1F36;letter-spacing:-0.03em;margin:0">
                Your Personal<br>Financial Advisor
            </h1>
            <p style="color:#6B7280;font-size:1rem;margin-top:0.8rem;font-weight:400;max-width:400px;margin-left:auto;margin-right:auto">
                Enter your financial details and get instant personalized insights,
                persona classification, and smart recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Feature pills
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:0.6rem;flex-wrap:wrap;margin-bottom:2.5rem">
        <div style="background:#EEF2FF;color:#4361EE;padding:0.4rem 1rem;border-radius:50px;font-size:0.8rem;font-weight:600">📊 Spending Analysis</div>
        <div style="background:#F0FDF4;color:#166534;padding:0.4rem 1rem;border-radius:50px;font-size:0.8rem;font-weight:600">🎯 Goal Feasibility</div>
        <div style="background:#FFFBEB;color:#92400E;padding:0.4rem 1rem;border-radius:50px;font-size:0.8rem;font-weight:600">💡 Smart Recommendations</div>
        <div style="background:#FEF3C7;color:#92400E;padding:0.4rem 1rem;border-radius:50px;font-size:0.8rem;font-weight:600">🤖 AI Persona Match</div>
    </div>
    """, unsafe_allow_html=True)

    # CTA
    if st.button("🚀  Get My Financial Analysis", key="start_btn"):
        st.session_state.page = "income"
        st.rerun()

    st.markdown("""
    <p style="text-align:center;color:#9CA3AF;font-size:0.78rem;margin-top:1rem">
        🔒 Your data stays on your device — nothing is stored or shared
    </p>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: INCOME & EXPENSES
# ─────────────────────────────────────────
def show_income():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-badge">Step 1 of 3 · Income & Expenses</div>
    <div class="page-title">What does your<br>monthly money look like?</div>
    <div class="page-sub">Enter your income and what you spend each month in Ksh</div>
    """, unsafe_allow_html=True)

    # Income inputs
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Monthly Income (Ksh)", min_value=0, value=50000, step=1000, key="income")
    with col2:
        savings = st.number_input("Monthly Savings (Ksh)", min_value=0, value=8000, step=500, key="savings")

    # Expenses inputs
    col3, col4 = st.columns(2)
    with col3:
        housing       = st.number_input("Housing / Rent (Ksh)",      min_value=0, value=12000, step=500,  key="housing")
        transport     = st.number_input("Transport (Ksh)",            min_value=0, value=4000,  step=500,  key="transport")
        entertainment = st.number_input("Entertainment (Ksh)",        min_value=0, value=3000,  step=500,  key="entertainment")
    with col4:
        food          = st.number_input("Food & Groceries (Ksh)",     min_value=0, value=8000,  step=500,  key="food")
        utilities     = st.number_input("Utilities (Ksh)",            min_value=0, value=2500,  step=500,  key="utilities")
        subscriptions = st.number_input("Subscriptions (Ksh)",        min_value=0, value=1500,  step=500,  key="subscriptions")

    total_exp = housing + food + transport + entertainment + utilities + subscriptions
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Continue  →", key="next_income"):
        st.session_state.form_data.update({
            "income": income, "savings": savings,
            "housing": housing, "food": food, "transport": transport,
            "entertainment": entertainment, "utilities": utilities,
            "subscriptions": subscriptions, "total_expenses": total_exp
        })
        st.session_state.page = "goal"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: GOAL
# ─────────────────────────────────────────
def show_goal():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-badge">Step 2 of 3 · Financial Goal</div>
    <div class="page-title">What are you<br>saving towards?</div>
    <div class="page-sub">Tell us about your financial goal so we can assess if it's achievable</div>
    """, unsafe_allow_html=True)

    goal_type = st.selectbox(
        "Goal Type",
        ["Emergency Fund", "Car", "House", "Travel", "Business", "Education"],
        key="goal_type"
    )

    col1, col2 = st.columns(2)
    with col1:
        target = st.number_input("Target Amount (Ksh)", min_value=1000, value=200000, step=5000, key="target")
    with col2:
        timeline = st.slider("Timeline (Months)", min_value=1, max_value=120, value=24, key="timeline")

    required_monthly = target / timeline
    income = st.session_state.form_data.get("income", 1)
    total_exp = st.session_state.form_data.get("total_expenses", 0)
    surplus = income - total_exp

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.8rem">
        <div style="background:#F0F4FF;border-radius:12px;padding:0.9rem 1.2rem;text-align:center">
            <div style="font-size:0.72rem;color:#4361EE;font-weight:700;text-transform:uppercase;letter-spacing:0.08em">Required Monthly</div>
            <div style="font-size:1.2rem;font-weight:800;color:#1A1F36;margin-top:0.2rem">Ksh {required_monthly:,.0f}</div>
        </div>
        <div style="background:#F0FDF4;border-radius:12px;padding:0.9rem 1.2rem;text-align:center">
            <div style="font-size:0.72rem;color:#166534;font-weight:700;text-transform:uppercase;letter-spacing:0.08em">Your Surplus</div>
            <div style="font-size:1.2rem;font-weight:800;color:#1A1F36;margin-top:0.2rem">Ksh {surplus:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("← Back", key="back_goal"):
            st.session_state.page = "income"
            st.rerun()
    with col_next:
        if st.button("Analyze My Finances →", key="next_goal"):
            st.session_state.form_data.update({
                "goal_type": goal_type, "target": target,
                "timeline": timeline, "required_monthly": required_monthly
            })
            st.session_state.page = "results"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────
def show_results():
    d = st.session_state.form_data
    income        = d["income"]
    savings       = d["savings"]
    total_exp     = d["total_expenses"]
    debt          = 0  # Assuming no debt for simplicity
    entertainment = d["entertainment"]
    subscriptions = d["subscriptions"]
    target        = d["target"]
    timeline      = d["timeline"]
    required      = d["required_monthly"]
    surplus       = income - total_exp

    savings_rate        = savings / income if income > 0 else 0
    debt_to_income      = debt / income if income > 0 else 0
    spending_ratio      = total_exp / income if income > 0 else 0
    discretionary_ratio = (entertainment + subscriptions) / income if income > 0 else 0
    subscription_ratio  = subscriptions / income if income > 0 else 0

    persona_name, persona_icon, persona_color, persona_bg, persona_desc = predict_persona(
        savings_rate, debt_to_income, spending_ratio,
        discretionary_ratio, subscription_ratio
    )
    feasibility, pill_class, progress = classify_feasibility(surplus, required)
    recs = get_recommendations(
        persona_name, feasibility, savings_rate,
        spending_ratio, discretionary_ratio, subscription_ratio
    )

    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="step-badge">Step 3 of 3 · Your Results</div>
    <div class="page-title">Your Financial<br>Analysis is Ready</div>
    <div class="page-sub">Here's a complete breakdown of your financial health</div>
    """, unsafe_allow_html=True)

    # ── Persona Banner ──
    st.markdown(f"""
    <div style="background:{persona_bg};border:2px solid {persona_color}22;border-radius:20px;padding:1.5rem 1.8rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:1.2rem">
        <div style="font-size:2.8rem">{persona_icon}</div>
        <div>
            <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:{persona_color};margin-bottom:0.2rem">Your Financial Persona</div>
            <div style="font-size:1.4rem;font-weight:800;color:#1A1F36;letter-spacing:-0.02em">{persona_name}</div>
            <div style="font-size:0.85rem;color:#6B7280;margin-top:0.2rem">{persona_desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-item">
            <div class="metric-label">💵 Monthly Income</div>
            <div class="metric-value">Ksh {income:,.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">🏦 Monthly Savings</div>
            <div class="metric-value">Ksh {savings:,.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">📊 Savings Rate</div>
            <div class="metric-value">{savings_rate*100:.1f}%</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">✨ Monthly Surplus</div>
            <div class="metric-value" style="color:{'#16A34A' if surplus > 0 else '#DC2626'}">Ksh {surplus:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Goal Feasibility ──
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Goal Feasibility</div>', unsafe_allow_html=True)
    pct = progress * 100
    prog_color = "#16A34A" if feasibility == "Feasible" else "#F59E0B" if feasibility == "Partially Feasible" else "#DC2626"

    st.markdown(f"""
    <div class="goal-row"><span class="gl">Goal Type</span><span class="gv">{d['goal_type']}</span></div>
    <div class="goal-row"><span class="gl">Target Amount</span><span class="gv">Ksh {target:,.0f}</span></div>
    <div class="goal-row"><span class="gl">Timeline</span><span class="gv">{timeline} months</span></div>
    <div class="goal-row"><span class="gl">Required Monthly Savings</span><span class="gv">Ksh {required:,.0f}</span></div>
    <div class="goal-row"><span class="gl">Your Monthly Surplus</span><span class="gv">Ksh {surplus:,.0f}</span></div>
    <div class="goal-row">
        <span class="gl">Feasibility Status</span>
        <span class="{pill_class} status-pill">{feasibility}</span>
    </div>
    <div style="margin-top:1rem">
        <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6B7280;margin-bottom:0.4rem">
            <span>Goal Capacity</span>
            <span style="color:{prog_color};font-weight:700">{pct:.1f}%</span>
        </div>
        <div class="prog-wrap">
            <div class="prog-fill" style="width:{pct}%;background:{prog_color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Spending Chart ──
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Your Spending Breakdown</div>', unsafe_allow_html=True)

    expense_cols = ["Housing", "Food", "Transport", "Entertainment", "Subscriptions", "Utilities"]
    values = [d["housing"], d["food"], d["transport"], d["entertainment"], d["subscriptions"], d["utilities"]]
    # App theme colors for donut chart (shades of blue)
    colors = ["#4361EE", "#5A67D8", "#7C3AED", "#9F7AEA", "#A78BFA", "#C4B5FD"]

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Create donut chart without percentages
    wedges, texts = ax.pie(values, labels=expense_cols, colors=colors, startangle=90, wedgeprops={'width': 0.6})

    # Add a white circle in the center
    centre_circle = plt.Circle((0,0), 0.3, fc='white')
    ax.add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    ax.set_title(f"Total Expenses: Ksh {sum(values):,.0f}", fontsize=10, color="#1A1F36", fontweight='700', pad=15)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">💡 Your Personalized Recommendations</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#6B7280;font-size:0.82rem;margin-bottom:1rem">{len(recs)} recommendations based on your financial profile</div>', unsafe_allow_html=True)

    card_map = {
        "success": "rec-success",
        "warn":    "rec-warn",
        "danger":  "rec-danger",
        "info":    "rec-info"
    }
    for rtype, icon, text in recs:
        st.markdown(f"""
        <div class="rec-item {card_map[rtype]}">
            <div class="rec-icon">{icon}</div>
            <div>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Start Over ──
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Adjust My Data", key="back_results"):
            st.session_state.page = "goal"
            st.rerun()
    with col2:
        if st.button("🔄 Start New Analysis", key="restart"):
            st.session_state.page = "home"
            st.session_state.form_data = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth()
else:
    render_sidebar()

    page = st.session_state.page
    if   page == "home":     show_home()
    elif page == "income":   show_income()
    elif page == "goal":     show_goal()
    elif page == "results":  show_results()
    elif page == "account":  show_account()
    elif page == "settings": show_settings()