"""
SentiScope AI — Deep Learning Powered Twitter Sentiment Analysis
==================================================================
Artificial Intelligence Project
Developed by Abhishek Panchal, Atharva Patil, Sumit Biradar, Sarthak Sasane

Run with:
    streamlit run app.py

Optional extra for PDF export:
    pip install fpdf2

Expected folder layout (same as the rest of the project):
    Sentiment-Analysis-Twitter/
        app.py                 <- this file
        models/
            lstm_model.keras
            tokenizer.pkl
        images/
            sentiment_distribution.png, wordcloud.png, ...
        processed_data/
            cleaned_tweets.csv
"""

import os
import re
import random
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="SentiScope AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# BRANDING & CONSTANTS
# ================================================================
APP_NAME = "SentiScope AI"
APP_TAGLINE = "Deep Learning Powered Twitter Sentiment Analysis"
APP_DESCRIPTION = "Analyze customer opinions using Artificial Intelligence and Natural Language Processing."

TEAM = ["Abhishek Panchal", "Atharva Patil", "Sumit Biradar", "Sarthak Sasane"]

MODELS_DIR = "models"
IMAGES_DIR = "images"
DATA_PATH = os.path.join("processed_data", "cleaned_tweets.csv")
MAX_LEN = 50
FALLBACK_DATASET_SIZE = 14640  # used only if cleaned_tweets.csv isn't found

LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
EMOJIS = {0: "😠", 1: "😐", 2: "😊"}
COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}
COLOR_BY_NAME = {"Negative": COLORS[0], "Neutral": COLORS[1], "Positive": COLORS[2]}
EMOJI_BY_NAME = {"Negative": EMOJIS[0], "Neutral": EMOJIS[1], "Positive": EMOJIS[2]}
REACTION_TEXT = {
    "Positive": "😊 Excellent!",
    "Neutral": "😐 Neutral Opinion",
    "Negative": "😠 Customer Complaint",
}

# NOTE: all three values are real accuracy_score results from the
# notebooks (01_Preprocessing_and_EDA.ipynb for LR/NB, 02_Deep_Learning_LSTM.ipynb
# for LSTM after the Bidirectional LSTM + class_weight retrain). LSTM went
# from a broken 62.94% (majority-class collapse) to a real, working 78%.
MODEL_ACCURACY = {
    "Logistic Regression": 0.78,
    "Naive Bayes": 0.73,
    "LSTM": 0.78,
}

SAMPLE_TWEETS = {
    "😊 Positive": "I absolutely loved the flight, the crew was amazing and so helpful!",
    "😠 Negative": "Worst airline experience ever, my flight was delayed for 5 hours.",
    "😐 Neutral": "My flight departs at 6 PM today from terminal 2.",
}

TECH_STACK = [
    "Python", "TensorFlow", "Streamlit", "Pandas",
    "NumPy", "NLTK", "Scikit-learn", "Matplotlib",
]

AI_FACTS = [
    "🐦 Roughly 500 million tweets are posted every day — about 6,000 every second.",
    "🤖 Automated sentiment analysis lets companies monitor customer satisfaction at a "
    "scale no human support team could match manually.",
]

LOADING_STEPS = [
    "🧠 Loading AI Model...",
    "🧹 Cleaning Tweet...",
    "🔤 Tokenizing...",
    "📐 Padding & Predicting...",
    "✨ Generating Insights...",
]

# ================================================================
# THEME (custom elements only — see Settings page for why)
# ================================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

if st.session_state.theme_mode == "light":
    CARD_BG, CARD_BORDER, TEXT_COLOR = "rgba(0,0,0,0.03)", "rgba(0,0,0,0.10)", "#1a1a1a"
    FEED_BG, FEED_BORDER = "rgba(0,0,0,0.02)", "rgba(0,0,0,0.15)"
else:
    CARD_BG, CARD_BORDER, TEXT_COLOR = "rgba(255,255,255,0.04)", "rgba(255,255,255,0.08)", "#e6edf3"
    FEED_BG, FEED_BORDER = "rgba(255,255,255,0.03)", "rgba(255,255,255,0.15)"

CSS_TEMPLATE = """
<style>
    .main-header {
        background: linear-gradient(90deg,#1f4e79,#0b84f3);
        padding: 28px 20px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .card {
        background: __CARD_BG__;
        border: 1px solid __CARD_BORDER__;
        color: __TEXT_COLOR__;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .result-card {
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin: 10px 0 20px 0;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .badge {
        display: inline-block;
        background: rgba(11,132,243,0.15);
        color: #6cb2f7;
        border: 1px solid rgba(11,132,243,0.35);
        padding: 6px 14px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .live-badge {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.4);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .feed-item {
        border-left: 3px solid __FEED_BORDER__;
        padding: 8px 14px;
        margin-bottom: 8px;
        background: __FEED_BG__;
        color: __TEXT_COLOR__;
        border-radius: 8px;
        font-size: 0.92rem;
    }
    .gauge-track {
        height: 26px;
        border-radius: 999px;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #22c55e);
        position: relative;
        margin: 6px 0 2px 0;
    }
    .gauge-marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 36px;
        background: white;
        border-radius: 2px;
        box-shadow: 0 0 6px rgba(0,0,0,0.5);
    }
    .workflow-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin: 12px 0 20px 0;
    }
    .workflow-step {
        background: __CARD_BG__;
        color: __TEXT_COLOR__;
        border: 1px solid __CARD_BORDER__;
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .workflow-arrow {
        color: #6cb2f7;
        font-size: 1.2rem;
    }
    .footer-box {
        text-align: center;
        color: #9aa4b2;
        padding: 18px 0 6px 0;
        font-size: 0.9rem;
    }
</style>
"""
css = (
    CSS_TEMPLATE
    .replace("__CARD_BG__", CARD_BG)
    .replace("__CARD_BORDER__", CARD_BORDER)
    .replace("__TEXT_COLOR__", TEXT_COLOR)
    .replace("__FEED_BG__", FEED_BG)
    .replace("__FEED_BORDER__", FEED_BORDER)
)
st.markdown(css, unsafe_allow_html=True)


# ================================================================
# CACHED LOADERS
# ================================================================
@st.cache_resource(show_spinner=False)
def load_lstm_model():
    path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_nlp_tools():
    return set(stopwords.words("english")), WordNetLemmatizer()


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


lstm_model = load_lstm_model()
tokenizer = load_tokenizer()
stop_words, lemmatizer = load_nlp_tools()
df = load_dataset()
model_ready = lstm_model is not None and tokenizer is not None


# ================================================================
# TEXT CLEANING + PREDICTION (mirrors the training pipeline)
# ================================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


def predict_sentiment(text: str):
    """Single-text prediction (no staged animation) — used where a quick
    result is needed without the step-by-step loading sequence."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    prediction = lstm_model.predict(padded, verbose=0)[0]
    predicted_class = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return predicted_class, confidence, prediction, cleaned, tokens


def predict_batch(texts):
    """Batched prediction — used by the Live Stream simulation."""
    cleaned_list = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned_list)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    preds = lstm_model.predict(padded, verbose=0)
    classes = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    return classes, confidences, preds, cleaned_list


def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"📁 `{path}` not found yet — run the notebook cell that generates it.")


def get_tweet_pool():
    if df is not None and "tweet" in df.columns:
        pool = df["tweet"].dropna().astype(str).tolist()
        if len(pool) > 0:
            return pool
    return list(SAMPLE_TWEETS.values()) * 20


def render_gauge(confidence: float, color: str):
    """Horizontal 'confidence meter' gauge — red-to-green track with a marker."""
    pct = max(0.0, min(1.0, confidence)) * 100
    st.markdown(f"""
    <div class='gauge-track'>
        <div class='gauge-marker' style='left: calc({pct}% - 2px);'></div>
    </div>
    <div style='text-align:center; font-size:1.6rem; font-weight:800; color:{color};'>
        {pct:.2f}%
    </div>
    """, unsafe_allow_html=True)


from fpdf import FPDF

def generate_pdf_report(tweet, cleaned, label, confidence, elapsed):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(190, 10, "SentiScope AI - Prediction Report", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)

        pdf.multi_cell(190, 8, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        pdf.multi_cell(190, 8, f"Original Tweet:\n{tweet}")
        pdf.multi_cell(190, 8, f"Cleaned Text:\n{cleaned or '(empty after cleaning)'}")
        pdf.multi_cell(190, 8, f"Predicted Sentiment: {label}")
        pdf.multi_cell(190, 8, f"Confidence: {confidence*100:.2f}%")
        pdf.multi_cell(190, 8, f"Prediction Time: {elapsed:.2f} sec")

        return bytes(pdf.output())

    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None

# ================================================================
# SESSION STATE
# ================================================================
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "stream_history" not in st.session_state:
    st.session_state.stream_history = []
if "stream_seeded" not in st.session_state:
    st.session_state.stream_seeded = False


# ================================================================
# LIVE STREAM HELPERS
# ================================================================
def seed_24h_history(n=150):
    pool = get_tweet_pool()
    n = min(n, len(pool))
    sampled = random.sample(pool, n)
    classes, confidences, _, _ = predict_batch(sampled)

    now = datetime.now()
    records = []
    for tweet, cls, conf in zip(sampled, classes, confidences):
        ts = now - timedelta(minutes=random.uniform(0, 24 * 60))
        records.append({
            "timestamp": ts,
            "tweet": tweet,
            "sentiment": LABELS[int(cls)],
            "confidence": float(conf),
        })
    records.sort(key=lambda r: r["timestamp"])
    st.session_state.stream_history = records
    st.session_state.stream_seeded = True


def run_live_simulation(n_tweets, delay):
    pool = get_tweet_pool()
    sampled = random.choices(pool, k=n_tweets)
    classes, confidences, _, _ = predict_batch(sampled)

    feed_placeholder = st.empty()
    progress = st.progress(0)

    for i, (tweet, cls, conf) in enumerate(zip(sampled, classes, confidences)):
        label = LABELS[int(cls)]
        record = {
            "timestamp": datetime.now(),
            "tweet": tweet,
            "sentiment": label,
            "confidence": float(conf),
        }
        st.session_state.stream_history.append(record)

        with feed_placeholder.container():
            st.markdown(
                f"<div class='feed-item'>{EMOJI_BY_NAME[label]} "
                f"<b style='color:{COLOR_BY_NAME[label]}'>{label}</b> "
                f"({conf * 100:.1f}%) — <i>{tweet}</i></div>",
                unsafe_allow_html=True,
            )

        progress.progress((i + 1) / n_tweets)
        time.sleep(delay)

    progress.empty()
    feed_placeholder.empty()


# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.title("📌 Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "✍️ Analyze Tweet",
        "📡 Live Stream",
        "📊 Dataset",
        "🧠 Models",
        "📚 About",
        "⚙️ Settings",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project")
st.sidebar.write("Twitter Sentiment Analysis")

st.sidebar.subheader("Dataset")
st.sidebar.info("Twitter US Airline Sentiment")

st.sidebar.subheader("Algorithm")
st.sidebar.success("LSTM Deep Learning")

st.sidebar.markdown("---")
st.sidebar.subheader("👨‍💻 Development Team")
for dev in TEAM:
    st.sidebar.write(f"• {dev}")

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Stats")
    st.sidebar.metric("Total Tweets", f"{len(df):,}")

st.sidebar.markdown("---")
if model_ready:
    st.sidebar.success("Ready for Prediction")
else:
    st.sidebar.error("Model / tokenizer not found in `models/`")


# ================================================================
# PAGE: HOME
# ================================================================
def render_home():
    st.markdown(f"""
    <div class='main-header'>
        <h1>💬 {APP_NAME}</h1>
        <h4>{APP_TAGLINE}</h4>
        <p>{APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<span class='badge'>🧠 LSTM Deep Learning</span>"
        "<span class='badge'>🐦 Twitter US Airline Sentiment</span>"
        "<span class='badge'>🎓 AI/ML Project</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    total_tweets = len(df) if df is not None else FALLBACK_DATASET_SIZE
    best_acc = max(MODEL_ACCURACY.values())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tweets", f"{total_tweets:,}")
    c2.metric("Models", "3")
    c3.metric("Best Accuracy", f"{best_acc * 100:.0f}%")
    c4.metric("Classes", "3")

    st.markdown(f"""
    <div class='card'>
        <h4>👋 Welcome to {APP_NAME}</h4>
        Analyze Twitter opinions using <b>Deep Learning (LSTM)</b><br><br>
        📊 <b>{total_tweets:,}</b> Tweets &nbsp;|&nbsp;
        🎯 <b>3</b> Sentiment Classes &nbsp;|&nbsp;
        ⚡ <b>Real-Time</b> Prediction
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h4>🎯 Objective</h4>"
                     "Classify airline-related tweets into Positive, Neutral, "
                     "or Negative sentiment using a trained LSTM model.</div>",
                     unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h4>📂 Dataset</h4>"
                     "Twitter US Airline Sentiment dataset — real "
                     "customer tweets about six US airlines.</div>",
                     unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h4>⚙️ Pipeline</h4>"
                     "Cleaning → Tokenization → Padding → LSTM → "
                     "Prediction, served live through this dashboard.</div>",
                     unsafe_allow_html=True)

    st.markdown("#### 📚 Project Workflow")
    steps = ["Dataset", "Cleaning", "EDA", "TF-IDF", "Classical ML", "LSTM", "Dashboard"]
    row_html = "<div class='workflow-row'>"
    for i, step in enumerate(steps):
        row_html += f"<div class='workflow-step'>{step}</div>"
        if i != len(steps) - 1:
            row_html += "<div class='workflow-arrow'>➜</div>"
    row_html += "</div>"
    st.markdown(row_html, unsafe_allow_html=True)

    st.info("👉 Try **✍️ Analyze Tweet** for a single prediction, or **📡 Live Stream** "
            "to see the simulated real-time feed and sentiment trend.")

    st.markdown("---")
    st.markdown("#### 💡 Did You Know?")
    fc1, fc2 = st.columns(2)
    for col, fact in zip([fc1, fc2], AI_FACTS):
        col.info(fact)


# ================================================================
# PAGE: ANALYZE TWEET
# ================================================================
def render_analyze():
    st.subheader("✍️ Enter a Tweet")

    st.caption("Try a sample tweet, or clear the box and write your own.")
    c1, c2, c3, c4 = st.columns(4)
    sample_cols = [c1, c2, c3]
    for col, (label, sample) in zip(sample_cols, SAMPLE_TWEETS.items()):
        if col.button(label, use_container_width=True):
            st.session_state.tweet_input = sample
    if c4.button("🧹 Clear", use_container_width=True):
        st.session_state.tweet_input = ""

    tweet = st.text_area(
        "✍️ Tweet Text",
        key="tweet_input",
        placeholder="Example: Flight was delayed for 3 hours...",
        height=150,
        max_chars=280,
        help="Enter any airline-related tweet to see its predicted sentiment.",
    )
    w1, w2 = st.columns(2)
    w1.caption(f"Characters: {len(tweet)}/280")
    w2.caption(f"Words: {len(tweet.split())}")

    predict = st.button("🚀 Predict Sentiment", use_container_width=True, type="primary")

    if predict:
        if not model_ready:
            st.error("Model or tokenizer isn't loaded — check the `models/` folder.")
        elif tweet.strip() == "":
            st.warning("Please enter a tweet.")
        else:
            status = st.empty()
            start_time = time.time()

            status.info(LOADING_STEPS[0])
            time.sleep(0.12)

            status.info(LOADING_STEPS[1])
            cleaned = clean_text(tweet)
            time.sleep(0.12)

            status.info(LOADING_STEPS[2])
            sequence = tokenizer.texts_to_sequences([cleaned])
            time.sleep(0.12)

            status.info(LOADING_STEPS[3])
            padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
            prediction = lstm_model.predict(padded, verbose=0)[0]
            time.sleep(0.12)

            status.info(LOADING_STEPS[4])
            time.sleep(0.12)
            status.empty()

            elapsed = time.time() - start_time
            pred_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            probs = prediction
            tokens = cleaned.split()

            label = LABELS[pred_class]
            emoji = EMOJIS[pred_class]
            color = COLORS[pred_class]

            st.markdown(f"""
            <div class='result-card' style='background:{color}22; border-color:{color};'>
                <h1 style='margin:0;'>{emoji}</h1>
                <h2 style='margin:6px 0; color:{color};'>{label}</h2>
                <p style='margin:0; opacity:0.85;'>{REACTION_TEXT[label]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🎚️ Confidence Meter")
            render_gauge(confidence, color)

            m1, m2 = st.columns(2)
            m1.metric("⚡ Prediction Time", f"{elapsed:.2f} sec")
            m2.metric("🎯 Confidence", f"{confidence * 100:.2f}%")

            st.markdown("##### 📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability": probs,
            })
            b1, b2 = st.columns(2)
            with b1:
                st.bar_chart(prob_df.set_index("Sentiment"))
            with b2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    probs,
                    labels=["Negative", "Neutral", "Positive"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=[COLORS[0], COLORS[1], COLORS[2]],
                )
                ax.set_title("Prediction Distribution")
                st.pyplot(fig)
                plt.close(fig)

            with st.expander("🧹 How this tweet was processed"):
                st.markdown(f"**Original:**  \n{tweet}")
                st.markdown("⬇️")
                st.markdown(f"**Cleaned:**  \n`{cleaned if cleaned else '(empty after cleaning)'}`")
                st.markdown("⬇️")
                st.markdown(f"**Tokens:**  \n`{tokens}`")
                st.markdown("⬇️")
                st.markdown(f"**Prediction:** {emoji} **{label}**")

            pdf_bytes = generate_pdf_report(tweet, cleaned, label, confidence, elapsed)
            if pdf_bytes:
                st.download_button(
                    "📄 Download Prediction Report (PDF)",
                    pdf_bytes,
                    f"sentiscope_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            else:
                st.caption("Install `fpdf2` (`pip install fpdf2`) to enable PDF report downloads.")

            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tweet": tweet,
                "cleaned_tweet": cleaned,
                "sentiment": label,
                "confidence": round(confidence * 100, 2),
            })

    if st.session_state.history:
        st.markdown("---")
        with st.expander(f"🕓 Prediction History ({len(st.session_state.history)})"):
            hist_df = pd.DataFrame(st.session_state.history)
            st.dataframe(hist_df, use_container_width=True)

            st.markdown("###### 📊 History Breakdown")
            hist_counts = (
                hist_df["sentiment"].value_counts()
                .reindex(["Positive", "Neutral", "Negative"])
                .fillna(0)
            )
            st.bar_chart(hist_counts)

            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ Download History as CSV",
                    hist_df.to_csv(index=False).encode("utf-8"),
                    "prediction_history.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with d2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()


# ================================================================
# PAGE: LIVE STREAM
# ================================================================
def render_live_stream():
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader("📡 Live Tweet Stream Simulation")
        st.caption("Simulates airline tweets arriving in real time and scores each one with the LSTM model.")
    with h2:
        st.markdown("<div style='text-align:right; padding-top:18px;'>"
                     "<span class='live-badge'>● LIVE DEMO</span></div>", unsafe_allow_html=True)

    if not model_ready:
        st.error("Model or tokenizer isn't loaded — check the `models/` folder.")
        return

    if not st.session_state.stream_seeded:
        with st.spinner("Generating the last 24 hours of simulated activity..."):
            seed_24h_history(n=150)

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    n_tweets = c1.slider("Tweets to stream", 5, 50, 15,
                          help="How many new simulated tweets to stream in this run.")
    delay = c2.slider("Speed (sec/tweet)", 0.1, 2.0, 0.4)
    start = c3.button("▶️ Start", use_container_width=True, type="primary")
    reset = c4.button("🔄 Reset", use_container_width=True)

    if reset:
        st.session_state.stream_history = []
        st.session_state.stream_seeded = False
        st.rerun()

    if start:
        st.markdown("##### 📰 Incoming")
        run_live_simulation(n_tweets, delay)
        st.rerun()

    hist = st.session_state.stream_history
    if not hist:
        st.info("No streamed data yet — click **Start** to begin.")
        return

    hist_df = pd.DataFrame(hist)

    st.markdown("---")
    st.markdown("#### 🥧 Real-Time Sentiment Distribution")
    counts = hist_df["sentiment"].value_counts()
    total = counts.sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tweets Streamed", f"{total:,}")
    m2.metric("😊 Positive", f"{counts.get('Positive', 0) / total * 100:.1f}%")
    m3.metric("😐 Neutral", f"{counts.get('Neutral', 0) / total * 100:.1f}%")
    m4.metric("😠 Negative", f"{counts.get('Negative', 0) / total * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(4, 4))
    labels_present = counts.index.tolist()
    ax.pie(
        counts.values,
        labels=labels_present,
        autopct="%1.1f%%",
        startangle=90,
        colors=[COLOR_BY_NAME[l] for l in labels_present],
    )
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### 📈 Sentiment Trend — Last 24 Hours")
    trend_df = hist_df.copy()
    trend_df["hour_bucket"] = pd.to_datetime(trend_df["timestamp"]).dt.floor("h")
    trend_pivot = (
        trend_df.groupby(["hour_bucket", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    )
    st.line_chart(trend_pivot, color=["#ef4444", "#f59e0b", "#22c55e"])

    st.markdown("#### 📰 Recent Tweets")
    recent = hist_df.sort_values("timestamp", ascending=False).head(10)
    for _, row in recent.iterrows():
        label = row["sentiment"]
        st.markdown(
            f"<div class='feed-item'>{EMOJI_BY_NAME[label]} "
            f"<b style='color:{COLOR_BY_NAME[label]}'>{label}</b> "
            f"({row['confidence'] * 100:.1f}%) · "
            f"{pd.to_datetime(row['timestamp']).strftime('%H:%M:%S')} — "
            f"<i>{row['tweet']}</i></div>",
            unsafe_allow_html=True,
        )


# ================================================================
# PAGE: DATASET OVERVIEW
# ================================================================
def render_dataset():
    st.subheader("📈 Dataset Overview")

    if df is None:
        st.error(f"Dataset not found at `{DATA_PATH}`. Run `01_Preprocessing_and_EDA.ipynb` first.")
        return

    sentiment_col = "sentiment" if "sentiment" in df.columns else df.columns[-1]
    counts = df[sentiment_col].str.lower().value_counts()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", f"{len(df):,} Tweets")
    c2.metric("😊 Positive", f"{counts.get('positive', 0):,}")
    c3.metric("😐 Neutral", f"{counts.get('neutral', 0):,}")
    c4.metric("😠 Negative", f"{counts.get('negative', 0):,}")

    st.markdown("#### 📊 Sentiment Distribution")
    show_image(os.path.join(IMAGES_DIR, "sentiment_distribution.png"))

    st.markdown("#### ☁️ Word Clouds")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.caption("Overall")
        show_image(os.path.join(IMAGES_DIR, "wordcloud.png"))
    with w2:
        st.caption("Positive Tweets")
        show_image(os.path.join(IMAGES_DIR, "positive_wordcloud.png"))
    with w3:
        st.caption("Negative Tweets")
        show_image(os.path.join(IMAGES_DIR, "negative_wordcloud.png"))

    with st.expander("🔍 Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)


# ================================================================
# PAGE: MODEL COMPARISON
# ================================================================
def render_models():
    st.subheader("🧠 Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "LSTM")
    c2.metric("Framework", "TensorFlow")
    c3.metric("Tokenizer", "Loaded" if tokenizer is not None else "Missing")
    c4.metric("Classes", "3")

    st.markdown("---")
    st.subheader("⚖️ Model Comparison")

    comp_df = pd.DataFrame({
        "Model": list(MODEL_ACCURACY.keys()),
        "Accuracy": list(MODEL_ACCURACY.values()),
    })
    best_model = comp_df.loc[comp_df["Accuracy"].idxmax(), "Model"]
    comp_df["Winner"] = comp_df["Model"].apply(lambda m: "⭐" if m == best_model else "")

    st.dataframe(
        comp_df.style.format({"Accuracy": "{:.2%}"})
        .highlight_max(subset=["Accuracy"], color="rgba(34,197,94,0.25)"),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Accuracy values are the real, measured test-set results from the notebooks — "
        "the ⭐ updates automatically to whichever model actually scores highest."
    )
    st.bar_chart(comp_df.set_index("Model")["Accuracy"])
    st.success(f"🏆 Best performing model: **{best_model}**")

    st.markdown("---")
    st.subheader("📉 Confusion Matrices")
    t1, t2, t3 = st.tabs(["Logistic Regression", "Naive Bayes", "LSTM"])
    with t1:
        show_image(os.path.join(IMAGES_DIR, "logistic_regression_confusion_matrix.png"))
    with t2:
        show_image(os.path.join(IMAGES_DIR, "naive_bayes_confusion_matrix.png"))
    with t3:
        show_image(os.path.join(IMAGES_DIR, "lstm_confusion_matrix.png"))

    st.markdown("---")
    st.subheader("📈 LSTM Training History")
    h1, h2 = st.columns(2)
    with h1:
        show_image(os.path.join(IMAGES_DIR, "lstm_accuracy.png"), "Training vs Validation Accuracy")
    with h2:
        show_image(os.path.join(IMAGES_DIR, "lstm_loss.png"), "Training vs Validation Loss")


# ================================================================
# PAGE: ABOUT
# ================================================================
def render_about():
    st.subheader("📚 About This Project")

    with st.expander("🎯 Objective", expanded=True):
        st.write(
            "Build an end-to-end sentiment analysis system that classifies airline-related "
            "tweets as Positive, Neutral, or Negative — from raw text cleaning through "
            "classical ML baselines to a deep learning (LSTM) model, served via this "
            "interactive dashboard."
        )
    with st.expander("📂 Dataset"):
        st.write(
            "**Twitter US Airline Sentiment** — 14,640 tweets about six major US airlines, "
            "each labeled Negative, Neutral, or Positive."
        )
    with st.expander("🧠 Algorithm"):
        st.write(
            "Tweets are cleaned (lowercasing, URL/mention/hashtag removal, stopword removal, "
            "lemmatization), tokenized, and padded before being fed into an LSTM network for "
            "3-class classification. Logistic Regression and Naive Bayes (on TF-IDF features) "
            "serve as classical baselines for comparison."
        )
        st.markdown("**LSTM Architecture**")
        arch_steps = ["Embedding", "LSTM", "Dropout", "Dense", "Softmax"]
        arch_html = "<div style='text-align:center;'>"
        for i, step in enumerate(arch_steps):
            arch_html += f"<span class='workflow-step'>{step}</span>"
            if i != len(arch_steps) - 1:
                arch_html += "<div class='workflow-arrow'>↓</div>"
        arch_html += "</div>"
        st.markdown(arch_html, unsafe_allow_html=True)
    with st.expander("💡 Applications"):
        st.write(
            "- Automated customer feedback triage for airlines\n"
            "- Brand and reputation monitoring on social media\n"
            "- Prioritizing urgent complaints for customer support teams\n"
            "- Aggregate sentiment trend tracking over time"
        )
    with st.expander("🔮 Future Scope"):
        st.write(
            "- Fine-tune a transformer model (e.g. BERT/DistilBERT) for higher accuracy\n"
            "- Add multilingual tweet support\n"
            "- Connect to a real Twitter/X stream instead of a simulated one\n"
            "- Expand beyond 3 classes to detect specific complaint categories"
        )

    st.markdown("---")
    st.markdown("#### 🛠️ Technologies Used")
    st.markdown("".join(f"<span class='badge'>{t}</span>" for t in TECH_STACK),
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📂 Project Structure")
    steps = ["Dataset", "Cleaning", "EDA", "TF-IDF", "Classical ML", "LSTM", "Dashboard"]
    row_html = "<div class='workflow-row'>"
    for i, step in enumerate(steps):
        row_html += f"<div class='workflow-step'>{step}</div>"
        if i != len(steps) - 1:
            row_html += "<div class='workflow-arrow'>➜</div>"
    row_html += "</div>"
    st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 👨‍💻 Development Team")
    st.markdown("".join(f"<span class='badge'>{t}</span>" for t in TEAM), unsafe_allow_html=True)

    team_lines = "<br>".join(TEAM)
    st.markdown(f"""
    <div class='footer-box'>
        Developed by<br><b>{team_lines}</b><br><br>
        Artificial Intelligence Project<br>
        © 2026
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# PAGE: SETTINGS
# ================================================================
def render_settings():
    st.subheader("⚙️ Settings")

    st.markdown("#### 🎨 Theme")
    st.caption(
        "Streamlit's own chrome (sidebar, buttons, page background) is fixed by "
        "`.streamlit/config.toml` at startup and can't be swapped live — that's a Streamlit "
        "platform limitation, not something this toggle can override. What this *does* "
        "control is the color scheme of this app's own custom cards, headers, and badges."
    )
    current_label = "🌙 Dark" if st.session_state.theme_mode == "dark" else "☀️ Light"
    choice = st.radio(
        "Custom element theme",
        ["🌙 Dark", "☀️ Light"],
        index=0 if st.session_state.theme_mode == "dark" else 1,
        horizontal=True,
    )
    new_mode = "dark" if choice == "🌙 Dark" else "light"
    if new_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = new_mode
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🗂️ Data & Cache")
    if st.button("🔄 Clear cached model/data and reload"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### ℹ️ Environment")
    e1, e2, e3 = st.columns(3)
    e1.metric("Model Loaded", "Yes" if model_ready else "No")
    e2.metric("Dataset Loaded", "Yes" if df is not None else "No")
    e3.metric("Current Theme", current_label)


# ================================================================
# ROUTING
# ================================================================
if page == "🏠 Home":
    render_home()
elif page == "✍️ Analyze Tweet":
    render_analyze()
elif page == "📡 Live Stream":
    render_live_stream()
elif page == "📊 Dataset":
    render_dataset()
elif page == "🧠 Models":
    render_models()
elif page == "📚 About":
    render_about()
elif page == "⚙️ Settings":
    render_settings()