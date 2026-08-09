import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Personal Financial Recommendation System",
    page_icon="💰",
    layout="wide"
)

# ─────────────────────────────────────────
# CSS — Inspired by User App Light Theme
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

/* Sidebar styling */
.sidebar .sidebar-content {
    background: #FFFFFF;
    border-radius: 20px;
    margin: 1rem;
    padding: 1rem;
    box-shadow: 0 2px 20px rgba(26,31,54,0.06);
}

/* Main content wrapper */
.main-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}

/* Cards */
.fin-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 20px rgba(26,31,54,0.06);
    border: 1px solid rgba(26,31,54,0.05);
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.metric-item {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
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

/* Subheaders */
h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1A1F36;
    margin-bottom: 1rem;
}

/* Info boxes */
.stAlert {
    border-radius: 14px !important;
    border: none !important;
    background: #EFF6FF !important;
    color: #1E40AF !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: #4361EE !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
}

/* Expander */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    border: 1px solid #E5E7EB !important;
}

hr {
    border-color: #F3F4F6 !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "financial_dataset.csv"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
KMEANS_PATH = BASE_DIR / "models" / "kmeans_model.pkl"

PERSONA_MAP = {
    0: "The Controlled Spender",
    1: "The Stretched Saver",
    2: "The Debt Burdened",
    3: "The High Risk Spender",
}


def classify_feasibility(row):
    if row["Monthly_Surplus"] <= 0:
        return "Not Feasible"
    ratio = row["Monthly_Surplus"] / row["Required_Monthly"]
    if ratio >= 1.0:
        return "Feasible"
    if ratio >= 0.6:
        return "Partially Feasible"
    return "Not Feasible"


def generate_recommendations(row):
    recommendations = []
    persona = row["Persona"]
    feasibility = row["Goal_Feasibility"]
    savings_rate = row["Savings_Rate"]
    spending_ratio = row["Spending_Ratio"]
    discretionary = row["Discretionary_Ratio"]
    subscription = row["Subscription_Ratio"]

    if persona == "The Controlled Spender":
        recommendations.append("✅ Your spending is well controlled — great financial discipline.")
        if savings_rate < 0.20:
            recommendations.append("💰 Try increasing your savings rate to at least 20% of your income.")
        else:
            recommendations.append("📈 Consider moving surplus savings into a money market or investment account.")
    elif persona == "The Stretched Saver":
        recommendations.append("⚠️ You save well but your total spending exceeds your income — this is unsustainable.")
        recommendations.append("🔍 Review your largest expense categories and identify areas to cut by 10-15%.")
        if discretionary > 0.15:
            recommendations.append("🎮 Your discretionary spending is high — set a strict monthly entertainment budget.")
    elif persona == "The Debt Burdened":
        recommendations.append("🚨 Your debt-to-income ratio is critically high — debt repayment must be your priority.")
        recommendations.append("💳 Focus on paying off high-interest debt first using the avalanche method.")
        recommendations.append("🚫 Avoid taking on any new debt until your ratio drops below 0.4.")
        if savings_rate < 0.10:
            recommendations.append("💰 Even saving 5-10% consistently will build a buffer against further debt.")
    elif persona == "The High Risk Spender":
        recommendations.append("🔴 Your finances are in a critical state — immediate corrective action is needed.")
        recommendations.append("📉 Your spending is nearly double your income — create a strict budget immediately.")
        if subscription > 0.10:
            recommendations.append("📱 You have very high subscription costs — audit and cancel all non-essential subscriptions.")
        recommendations.append("🏦 Speak with a financial advisor about debt restructuring options.")

    if feasibility == "Feasible":
        recommendations.append("🏆 Your goal is achievable within your timeline — stay consistent!")
        if savings_rate > 0.50:
            recommendations.append("🌟 Excellent savings rate! You may achieve your goal ahead of schedule.")
    elif feasibility == "Partially Feasible":
        recommendations.append("⏳ Your goal is partially achievable — you're about 60-99% of the way there.")
        recommendations.append("📊 Reducing monthly expenses by 10% could make your goal fully feasible.")
        recommendations.append("📅 Consider extending your goal timeline by 3-6 months as an alternative.")
    else:
        recommendations.append("❌ Your goal is not achievable at your current savings pace.")
        recommendations.append("🔄 Options: increase income, reduce expenses, lower target amount, or extend timeline.")
        if spending_ratio > 1.0:
            recommendations.append("⚡ Critical: You are spending more than you earn — address this before pursuing any goal.")

    return recommendations


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find data file: {DATA_PATH}")
    if not SCALER_PATH.exists() or not KMEANS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model files: {SCALER_PATH} and/or {KMEANS_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df = df[df["Savings_Rate"] <= 1.0].reset_index(drop=True)
    df["Spending_Ratio"] = df["Total_Expenses"] / df["Monthly_Income"]
    df["Discretionary_Ratio"] = (df["Entertainment"] + df["Subscriptions"]) / df["Monthly_Income"]
    df["Subscription_Ratio"] = df["Subscriptions"] / df["Monthly_Income"]
    df["Monthly_Surplus"] = df["Monthly_Income"] - df["Total_Expenses"]
    df["Required_Monthly"] = df["Target_Amount"] / df["Timeline_Months"]

    scaler = joblib.load(SCALER_PATH)
    km = joblib.load(KMEANS_PATH)

    features = [
        "Savings_Rate",
        "Debt_to_Income",
        "Spending_Ratio",
        "Discretionary_Ratio",
        "Subscription_Ratio",
    ]

    X = scaler.transform(df[features])
    df["Cluster"] = km.predict(X)
    df["Persona"] = df["Cluster"].map(PERSONA_MAP)
    df["Goal_Feasibility"] = df.apply(classify_feasibility, axis=1)
    df["Recommendations"] = df.apply(generate_recommendations, axis=1)
    return df


def render_dashboard(df):
    st.sidebar.image("https://img.icons8.com/fluency/96/money.png", width=80)
    st.sidebar.title("💰 Financial Planner")
    st.sidebar.markdown("---")

    user_id = st.sidebar.selectbox("Select User ID", df["User_ID"].tolist())
    user = df[df["User_ID"] == user_id].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Persona:** {user['Persona']}")
    st.sidebar.markdown(f"**Goal:** {user['Goal_Type']}")
    st.sidebar.markdown(f"**Feasibility:** {user['Goal_Feasibility']}")

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("💰 Personal Financial Recommendation System")
    st.markdown(f"### 👤 User {user_id} — {user['Persona']}")
    st.markdown("---")

    # Metrics in grid
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="metric-item">
        <div class="metric-label">💵 Monthly Income</div>
        <div class="metric-value">Ksh {user['Monthly_Income']:,.0f}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">🏦 Monthly Savings</div>
        <div class="metric-value">Ksh {user['Monthly_Savings']:,.0f}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">📊 Savings Rate</div>
        <div class="metric-value">{user['Savings_Rate']*100:.1f}%</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">💳 Debt-to-Income</div>
        <div class="metric-value">{user['Debt_to_Income']:.2f}</div>
    </div>
    <div class="metric-item">
        <div class="metric-label">✨ Monthly Surplus</div>
        <div class="metric-value">Ksh {user['Monthly_Surplus']:,.0f}</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.subheader("📊 Spending Breakdown")
        expense_cols = ["Housing", "Food", "Transport", "Entertainment", "Subscriptions", "Utilities"]
        values = [user[col] for col in expense_cols]

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(values, labels=expense_cols, startangle=90, wedgeprops={'width': 0.6})
        ax1.set_title(f"Expense Distribution — User {user_id}")
        st.pyplot(fig1)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.subheader("🎯 Goal Summary")
        feasibility_color = {
            "Feasible": "green",
            "Partially Feasible": "orange",
            "Not Feasible": "red",
        }
        color = feasibility_color[user["Goal_Feasibility"]]

        st.markdown(f"**Goal Type:** {user['Goal_Type']}")
        st.markdown(f"**Target Amount:** Ksh {user['Target_Amount']:,.0f}")
        st.markdown(f"**Timeline:** {user['Timeline_Months']} months")
        st.markdown(f"**Required Monthly Savings:** Ksh {user['Required_Monthly']:,.0f}")
        st.markdown(f"**Current Monthly Surplus:** Ksh {user['Monthly_Surplus']:,.0f}")
        st.markdown(f"**Feasibility Status:** :{color}[**{user['Goal_Feasibility']}**]")

        if user["Required_Monthly"] > 0:
            progress = min(user["Monthly_Surplus"] / user["Required_Monthly"], 1.0)
            progress = max(progress, 0.0)
            st.markdown("**Goal Progress Capacity:**")
            st.progress(progress)
            st.caption(f"{progress*100:.1f}% of required monthly savings covered")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="fin-card">', unsafe_allow_html=True)
    st.subheader("💡 Personalized Recommendations")
    for rec in user["Recommendations"]:
        st.info(rec)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Dataset Overview")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown("**Persona Distribution**")
        persona_counts = df["Persona"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        persona_counts.plot(kind="bar", ax=ax2, color="steelblue", edgecolor="black")
        ax2.set_ylabel("Number of Users")
        ax2.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="fin-card">', unsafe_allow_html=True)
        st.markdown("**Goal Feasibility Distribution**")
        feasibility_counts = df["Goal_Feasibility"].value_counts()
        colors = ["steelblue", "coral", "orange"]
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        feasibility_counts.plot(kind="bar", ax=ax3, color=colors, edgecolor="black")
        ax3.set_ylabel("Number of Users")
        ax3.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📂 View Full Dataset"):
        st.dataframe(
            df[
                [
                    "User_ID",
                    "Monthly_Income",
                    "Monthly_Savings",
                    "Total_Expenses",
                    "Savings_Rate",
                    "Debt_to_Income",
                    "Spending_Ratio",
                    "Persona",
                    "Goal_Type",
                    "Goal_Feasibility",
                    "Monthly_Surplus",
                ]
            ]
        )

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    try:
        df = load_data()
    except Exception as exc:
        st.error(f"Error loading app data: {exc}")
        st.stop()

    render_dashboard(df)


if __name__ == "__main__":
    main()