import os
import random
import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from preprocess import clean_text, detect_sarcasm

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Tweet Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_files()

# ---------------- LOAD DATASET IF AVAILABLE ----------------
@st.cache_data
def load_data():
    possible_files = ["tweet_sentiment.csv", "dataset.csv"]

    for file_name in possible_files:
        if os.path.exists(file_name):
            df = pd.read_csv(file_name, encoding="latin-1", header=None)
            df.columns = ["sentiment", "id", "date", "query", "user", "text"]

            df = df[["sentiment", "date", "user", "text"]].copy()
            df["sentiment"] = df["sentiment"].replace({
                0: "negative",
                4: "positive"
            })
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["text"] = df["text"].astype(str)
            df = df.dropna(subset=["date"]).copy()
            return df

    return pd.DataFrame(columns=["sentiment", "date", "user", "text"])

df = load_data()
dataset_available = not df.empty

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "sample_text" not in st.session_state:
    st.session_state["sample_text"] = ""

# ---------------- SAMPLE TWEETS ----------------
sample_tweets = [
    "I absolutely love this update, it works perfectly.",
    "This app is so slow and frustrating.",
    "Wow great job, the app crashed again.",
    "Amazing update, now nothing works.",
    "I like the design, but the app is still very annoying."
]

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top right, #182848 0%, #0b1020 45%, #070b14 100%);
        color: #f3f6fb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1426 0%, #0a1020 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] * {
        color: #e8eef9 !important;
    }

    .hero {
        background: linear-gradient(90deg, rgba(17,29,63,0.95), rgba(11,19,43,0.92));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 26px 28px;
        margin-bottom: 18px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.30);
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        line-height: 1.1;
    }

    .hero-sub {
        font-size: 17px;
        color: #b7c2d8;
    }

    .card {
        background: linear-gradient(180deg, rgba(17,24,39,0.92), rgba(10,15,28,0.94));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    .feature-card {
        background: linear-gradient(180deg, rgba(16,22,39,0.95), rgba(11,17,31,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        min-height: 138px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.20);
    }

    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #f4f7fc;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 14px;
        color: #aeb9cf;
        line-height: 1.6;
    }

    .section-title {
        font-size: 28px;
        font-weight: 800;
        color: #f4f7fc;
        margin-bottom: 8px;
    }

    .section-sub {
        font-size: 15px;
        color: #aeb9cf;
        margin-bottom: 16px;
    }

    .result-positive {
        background: linear-gradient(90deg, rgba(29,78,59,0.95), rgba(10,56,56,0.95));
        border: 1px solid rgba(52,211,153,0.45);
        border-left: 8px solid #22c55e;
        border-radius: 18px;
        padding: 18px;
        color: #ddfff0;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 12px;
        box-shadow: 0 0 18px rgba(34,197,94,0.18);
    }

    .result-negative {
        background: linear-gradient(90deg, rgba(90,24,32,0.95), rgba(58,19,25,0.95));
        border: 1px solid rgba(248,113,113,0.45);
        border-left: 8px solid #ef4444;
        border-radius: 18px;
        padding: 18px;
        color: #ffe8e8;
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 12px;
        box-shadow: 0 0 18px rgba(239,68,68,0.16);
    }

    .sarcasm-badge {
        display: inline-block;
        background: rgba(245, 158, 11, 0.16);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #ffd48a;
        font-weight: 700;
        font-size: 14px;
        padding: 8px 14px;
        border-radius: 999px;
        margin-bottom: 14px;
    }

    .metric-box {
        background: linear-gradient(180deg, rgba(17,24,39,0.92), rgba(12,17,30,0.95));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 18px rgba(0,0,0,0.22);
    }

    .metric-label {
        color: #9fb0cf;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
    }

    .history-item {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 10px;
        color: #dce5f5;
    }

    .footer-note {
        text-align: center;
        color: #8ea0c0;
        font-size: 13px;
        margin-top: 18px;
        padding-bottom: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.03) !important;
        color: #f4f7fc !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 8px 16px;
        color: #dce5f5;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1d4ed8, #2563eb) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ✈️ Tweet Sentiment")
    st.markdown("### Dashboard")
    page = st.radio(
        "Navigation",
        ["Predict", "Dashboard", "History"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### About")
    st.write("Professional sentiment analysis interface for tweet-style text using NLP and machine learning.")

    if dataset_available:
        st.markdown("### Dataset")
        st.caption("Analytics enabled")
    else:
        st.markdown("### Dataset")
        st.caption("Prediction-only mode")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="hero-title">Tweet Sentiment Dashboard</div>
    <div class="hero-sub">
        AI-powered sentiment analysis for tweets. Predict tone, inspect confidence, and flag sarcasm through a clean professional interface.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- FEATURE CARDS ----------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:28px;">🧠</div>
        <div class="feature-title">AI Prediction</div>
        <div class="feature-text">Predicts positive or negative sentiment using a trained machine learning model.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:28px;">🙂</div>
        <div class="feature-title">Sarcasm Detection</div>
        <div class="feature-text">Detects sarcastic patterns and adjusts the displayed result where needed.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:28px;">📊</div>
        <div class="feature-title">Confidence Scores</div>
        <div class="feature-text">Displays model confidence for each class through a visual prediction chart.</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="feature-card">
        <div style="font-size:28px;">⚡</div>
        <div class="feature-title">Quick Testing</div>
        <div class="feature-text">Use sample tweets, random examples, and session history for faster testing.</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------- PREDICT PAGE ----------------
if page == "Predict":
    left_col, right_col = st.columns([1.7, 1])

    with left_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analyze a Tweet</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Enter your own tweet or use one of the examples below.</div>', unsafe_allow_html=True)

        selected_sample = st.selectbox(
            "Choose a sample tweet",
            ["None"] + sample_tweets
        )

        btn1, btn2 = st.columns(2)

        if btn1.button("Random Example"):
            st.session_state["sample_text"] = random.choice(sample_tweets)
            st.rerun()

        if btn2.button("Clear Text"):
            st.session_state["sample_text"] = ""
            st.rerun()

        if selected_sample != "None":
            st.session_state["sample_text"] = selected_sample

        user_input = st.text_area(
            "Tweet text",
            value=st.session_state["sample_text"],
            height=180,
            placeholder="Type a tweet here..."
        )

        predict_clicked = st.button("Predict Sentiment")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Quick Tips</div>', unsafe_allow_html=True)
        st.markdown("""
- Use clear emotional words like **love**, **hate**, **great**, **annoying**
- Sarcasm often combines positive words with negative context
- Short tweet-style text usually works best
- Mixed-language text may work, but English is more reliable
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        if user_input.strip():
            cleaned_text = clean_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])

            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]
            is_sarcasm = detect_sarcasm(user_input)

            if is_sarcasm and prediction == "positive":
                prediction = "negative"

            if prediction == "positive":
                st.markdown(
                    '<div class="result-positive">😊 Predicted Sentiment: Positive</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-negative">😠 Predicted Sentiment: Negative</div>',
                    unsafe_allow_html=True
                )

            if is_sarcasm:
                st.markdown(
                    '<div class="sarcasm-badge">😏 Sarcasm detected</div>',
                    unsafe_allow_html=True
                )

            st.session_state["history"].insert(0, {
                "text": user_input,
                "prediction": prediction.capitalize(),
                "sarcasm": "Yes" if is_sarcasm else "No"
            })
            st.session_state["history"] = st.session_state["history"][:10]

            r1, r2 = st.columns([1.1, 1])

            with r1:
                prob_df = pd.DataFrame({
                    "Sentiment": model.classes_,
                    "Confidence": probabilities
                })

                fig_bar = px.bar(
                    prob_df,
                    x="Sentiment",
                    y="Confidence",
                    text_auto=".2f"
                )
                fig_bar.update_layout(
                    height=320,
                    xaxis_title="Sentiment",
                    yaxis_title="Confidence",
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e8eef9")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with r2:
                st.markdown("### Model Interpretation")
                if prediction == "positive":
                    st.write("The tweet contains stronger positive cues than negative ones.")
                else:
                    st.write("The tweet contains stronger negative cues or complaint-like wording.")

                if is_sarcasm:
                    st.write("A sarcastic pattern was detected, so the result was adjusted to better reflect the intended tone.")

                st.write("**Cleaned text used by the model:**")
                st.code(cleaned_text)
        else:
            st.warning("Please enter some text before predicting.")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DASHBOARD PAGE ----------------
elif page == "Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analytics Overview</div>', unsafe_allow_html=True)

    if dataset_available:
        total_tweets = len(df)
        positive_count = (df["sentiment"] == "positive").sum()
        negative_count = (df["sentiment"] == "negative").sum()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Total Tweets</div>
                <div class="metric-value">{total_tweets:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Positive</div>
                <div class="metric-value">{positive_count:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Negative</div>
                <div class="metric-value">{negative_count:,}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        g1, g2 = st.columns(2)

        with g1:
            sentiment_counts = df["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_donut = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.65
            )
            fig_donut.update_layout(
                height=340,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eef9")
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with g2:
            trend_df = df.copy()
            trend_df["day"] = trend_df["date"].dt.date
            trend_counts = trend_df.groupby(["day", "sentiment"]).size().reset_index(name="count")

            fig_line = px.line(
                trend_counts,
                x="day",
                y="count",
                color="sentiment",
                markers=True
            )
            fig_line.update_layout(
                height=340,
                xaxis_title="Date",
                yaxis_title="Tweet Count",
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eef9")
            )
            st.plotly_chart(fig_line, use_container_width=True)

        sample_df = df[["date", "user", "text", "sentiment"]].copy()
        sample_df["date"] = sample_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(sample_df.head(15), use_container_width=True)
    else:
        preview_df = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [62, 38]
        })

        g1, g2 = st.columns(2)

        with g1:
            fig_preview = px.pie(
                preview_df,
                names="Sentiment",
                values="Count",
                hole=0.65
            )
            fig_preview.update_layout(
                height=320,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eef9")
            )
            st.plotly_chart(fig_preview, use_container_width=True)

        with g2:
            preview_trend = pd.DataFrame({
                "Day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                "Positive": [12, 16, 11, 17, 15],
                "Negative": [8, 7, 10, 6, 9]
            })
            preview_long = preview_trend.melt(id_vars="Day", var_name="Sentiment", value_name="Count")

            fig_line_preview = px.line(
                preview_long,
                x="Day",
                y="Count",
                color="Sentiment",
                markers=True
            )
            fig_line_preview.update_layout(
                height=320,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eef9")
            )
            st.plotly_chart(fig_line_preview, use_container_width=True)

        st.info("Preview analytics shown. Add the dataset file later to enable full live dashboard analytics.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- HISTORY PAGE ----------------
else:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recent Prediction History</div>', unsafe_allow_html=True)

    if st.session_state["history"]:
        for item in st.session_state["history"]:
            st.markdown(f"""
            <div class="history-item">
                <b>Tweet:</b> {item['text']}<br>
                <b>Prediction:</b> {item['prediction']}<br>
                <b>Sarcasm:</b> {item['sarcasm']}
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No predictions yet. Use the Predict page to analyze a tweet first.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer-note">Built with Streamlit, NLP, and machine learning for professional tweet sentiment analysis.</div>',
    unsafe_allow_html=True
)