import json
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# -----------------------------------------------------------------------------
# Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis | LSTM Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# File Paths & Model Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent if (BASE_DIR.parent / "Models").exists() else BASE_DIR

MODEL_PATH = ROOT_DIR / "Models" / "sentiment_lstm_model.keras"
WORD_INDEX_PATH = ROOT_DIR / "Models" / "word_index.json"
CONFUSION_MATRIX_PATH = ROOT_DIR / "Results" / "confusion_matrix.png"

VOCAB_SIZE = 10000
MAX_LENGTH = 250

# -----------------------------------------------------------------------------
# Benchmark Sample Reviews
# -----------------------------------------------------------------------------
SAMPLE_REVIEWS = {
    "Select a pre-loaded sample...": "",
    "Positive Sample: Masterpiece Review": (
        "An extraordinary film with captivating storytelling, magnificent direction, "
        "and brilliant performances throughout. A true masterpiece of modern cinema."
    ),
    "Negative Sample: Critical Review": (
        "A completely uninspired and tedious waste of time. Poor acting, "
        "predictable plot, awful dialogue, and zero emotional resonance."
    ),
    "Nuanced Sample: Mixed Review": (
        "Visually impressive cinematography and stellar sound design, but "
        "the convoluted script and uneven pacing leave much to be desired."
    ),
}

# -----------------------------------------------------------------------------
# Vibrant & Professional Color Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Typography & Inputs */
    .stTextArea textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 0.95rem;
        line-height: 1.5;
        border-radius: 8px;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextArea textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    /* Buttons */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        color: #6366f1 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        background: rgba(99, 102, 241, 0.04) !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: #4f46e5 !important;
        background: rgba(99, 102, 241, 0.1) !important;
        color: #4338ca !important;
    }

    /* Header Container */
    .header-banner {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.10) 0%, rgba(59, 130, 246, 0.08) 50%, rgba(6, 182, 212, 0.06) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.5rem;
    }

    .header-title {
        font-size: 1.75rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        background: linear-gradient(135deg, #4f46e5 0%, #2563eb 60%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .header-subtitle {
        font-size: 0.92rem;
        opacity: 0.8;
        margin: 0.3rem 0 0 0;
        line-height: 1.4;
    }

    .status-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.85rem;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #10b981;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Section Headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .section-indicator {
        width: 4px;
        height: 16px;
        background: linear-gradient(180deg, #4f46e5, #06b6d4);
        border-radius: 2px;
        display: inline-block;
    }

    /* Verdict Card - Positive */
    .verdict-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.12);
        border-radius: 10px;
        padding: 1.35rem;
        margin-bottom: 1.15rem;
    }

    .status-badge-pos {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #ffffff;
        background: #10b981;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);
        margin-bottom: 0.5rem;
    }

    /* Verdict Card - Negative */
    .verdict-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.12);
        border-radius: 10px;
        padding: 1.35rem;
        margin-bottom: 1.15rem;
    }

    .status-badge-neg {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #ffffff;
        background: #ef4444;
        padding: 0.22rem 0.65rem;
        border-radius: 9999px;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.35);
        margin-bottom: 0.5rem;
    }

    /* Distribution Progress Bar Container */
    .dist-box {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1.15rem;
    }

    /* Metric Cards - 4 Vibrant Variations */
    .metric-card-indigo {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.10) 0%, rgba(99, 102, 241, 0.03) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .metric-val-indigo {
        color: #6366f1;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .metric-card-cyan {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.10) 0%, rgba(6, 182, 212, 0.03) 100%);
        border: 1px solid rgba(6, 182, 212, 0.25);
        border-left: 4px solid #06b6d4;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .metric-val-cyan {
        color: #06b6d4;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .metric-card-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.10) 0%, rgba(245, 158, 11, 0.03) 100%);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .metric-val-amber {
        color: #f59e0b;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .metric-card-purple {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.10) 0%, rgba(168, 85, 247, 0.03) 100%);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-left: 4px solid #a855f7;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.65rem;
    }
    .metric-val-purple {
        color: #a855f7;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .stat-label-themed {
        font-size: 0.76rem;
        font-weight: 600;
        opacity: 0.75;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    /* Awaiting Input Card */
    .awaiting-card {
        background: radial-gradient(circle at 50% 25%, rgba(99, 102, 241, 0.08) 0%, rgba(99, 102, 241, 0.02) 100%);
        border: 1.5px dashed rgba(99, 102, 241, 0.35);
        border-radius: 10px;
        padding: 3rem 1.5rem;
        text-align: center;
    }

    .awaiting-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(99, 102, 241, 0.12);
        border-radius: 9999px;
        color: #6366f1;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }

    /* Sidebar Cards */
    .sidebar-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(59, 130, 246, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.85rem;
    }

    .sidebar-stat-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #4f46e5;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model & Index Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_path: Path):
    """Load model with resilient fallback if version serialization requires reconstruction."""
    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
        model = Sequential([
            Input(shape=(MAX_LENGTH,)),
            Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
        model.load_weights(str(model_path))
        return model

@st.cache_data(show_spinner=False)
def load_word_index(index_path: Path):
    """Load vocabulary dictionary."""
    with open(str(index_path), "r", encoding="utf-8") as file:
        return json.load(file)

# -----------------------------------------------------------------------------
# Preprocessing Pipeline
# -----------------------------------------------------------------------------
def preprocess_text(text: str, word_index: dict):
    """
    Transforms raw input text into a padded integer sequence matching IMDB conventions:
    Index offset of +3 accounts for reserved tokens (0: pad, 1: start, 2: OOV).
    """
    words = text_to_word_sequence(text)
    sequence = [1]
    oov_count = 0

    for word in words:
        index = word_index.get(word, 2) + 3
        if index >= VOCAB_SIZE:
            index = 2
            oov_count += 1
        sequence.append(index)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return padded, len(words), oov_count

# Load resources
model = load_sentiment_model(MODEL_PATH)
word_index = load_word_index(WORD_INDEX_PATH)

# -----------------------------------------------------------------------------
# Sidebar: System Specifications & Diagnostics
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Model Specifications")
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="font-weight: 700; font-size: 0.92rem; color: #4f46e5; margin-bottom: 0.35rem;">
                Recurrent Neural Network
            </div>
            <div style="font-size: 0.85rem; opacity: 0.85; line-height: 1.45;">
                Trained on the IMDB 50k benchmark corpus for binary sentiment classification with sequence modeling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("#### Hyperparameters")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric(label="Vocab Limit", value=f"{VOCAB_SIZE:,}")
        st.metric(label="Embedding Dim", value="64")
    with col_sb2:
        st.metric(label="Max Sequence", value=f"{MAX_LENGTH}")
        st.metric(label="LSTM Units", value="64")

    st.divider()

    st.markdown("#### Test Evaluation")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Accuracy", value="85.87%")
    with col_m2:
        st.metric(label="Loss", value="0.3595")

    if CONFUSION_MATRIX_PATH.exists():
        with st.expander("Confusion Matrix"):
            st.image(
                str(CONFUSION_MATRIX_PATH),
                caption="Test Evaluation Matrix (IMDB Test Set)",
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# Main Header with Brand Gradient
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-banner">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;">
            <div>
                <h1 class="header-title">Movie Review Sentiment Analysis</h1>
                <p class="header-subtitle">
                    Natural language classification powered by a sequence-aware LSTM recurrent neural network.
                </p>
            </div>
            <div class="status-pill-online">
                <span class="pulse-dot"></span>
                Model Online
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

def handle_sample_selection():
    selected = st.session_state.get("sample_dropdown")
    if selected and SAMPLE_REVIEWS.get(selected):
        st.session_state["review_input"] = SAMPLE_REVIEWS[selected]

def handle_clear_text():
    st.session_state["review_input"] = ""
    st.session_state["analysis_result"] = None

# -----------------------------------------------------------------------------
# Main Layout: Two Balanced Columns
# -----------------------------------------------------------------------------
col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown(
        """
        <div class="section-title">
            <span class="section-indicator"></span>
            Review Input
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        label="Load an evaluation sample:",
        options=list(SAMPLE_REVIEWS.keys()),
        key="sample_dropdown",
        on_change=handle_sample_selection,
        label_visibility="collapsed"
    )

    current_text = st.text_area(
        label="Review text area",
        key="review_input",
        height=190,
        placeholder="Enter a movie review to analyze sentiment...",
        label_visibility="collapsed"
    )

    # Word and character count indicator
    words_entered = len(current_text.split()) if current_text.strip() else 0
    chars_entered = len(current_text)
    st.markdown(
        f"""
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem;">
            <span style="background: rgba(99, 102, 241, 0.08); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.8rem; font-weight: 600;">
                {words_entered} words
            </span>
            <span style="background: rgba(6, 182, 212, 0.08); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.2); border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.8rem; font-weight: 600;">
                {chars_entered} characters
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("Clear", on_click=handle_clear_text, use_container_width=True)

    if analyze_clicked:
        if current_text.strip():
            with st.spinner("Processing text and running LSTM inference..."):
                padded_seq, total_words, oov_words = preprocess_text(current_text, word_index)
                raw_prediction = float(model.predict(padded_seq, verbose=0)[0][0])
                st.session_state["analysis_result"] = {
                    "raw_score": raw_prediction,
                    "total_words": total_words,
                    "oov_words": oov_words,
                }
        else:
            st.session_state["analysis_result"] = None
            st.warning("Please enter or select a review to analyze.")

with col_output:
    st.markdown(
        """
        <div class="section-title">
            <span class="section-indicator"></span>
            Classification Output
        </div>
        """,
        unsafe_allow_html=True,
    )

    result = st.session_state["analysis_result"]

    if result is not None:
        score = result["raw_score"]
        is_positive = score >= 0.5

        pos_pct = score * 100
        neg_pct = (1 - score) * 100

        # Confidence level calculation
        confidence = pos_pct if is_positive else neg_pct
        if confidence >= 80:
            confidence_tier = "High Confidence"
        elif confidence >= 65:
            confidence_tier = "Moderate Confidence"
        else:
            confidence_tier = "Marginal Confidence"

        # Verdict card
        if is_positive:
            st.markdown(
                f"""
                <div class="verdict-positive">
                    <span class="status-badge-pos">Positive Sentiment</span>
                    <h3 style="margin: 0 0 0.35rem 0; color: #10b981; font-size: 1.6rem; font-weight: 800;">
                        Positive Review
                    </h3>
                    <div style="font-size: 0.92rem; opacity: 0.9; font-weight: 500;">
                        {confidence_tier} &bull; <strong style="color: #10b981;">{confidence:.1f}%</strong> positive probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-negative">
                    <span class="status-badge-neg">Negative Sentiment</span>
                    <h3 style="margin: 0 0 0.35rem 0; color: #ef4444; font-size: 1.6rem; font-weight: 800;">
                        Negative Review
                    </h3>
                    <div style="font-size: 0.92rem; opacity: 0.9; font-weight: 500;">
                        {confidence_tier} &bull; <strong style="color: #ef4444;">{confidence:.1f}%</strong> negative probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # High-contrast dual distribution bar
        st.markdown(
            f"""
            <div class="dist-box">
                <div style="display: flex; justify-content: space-between; font-size: 0.88rem; font-weight: 700; margin-bottom: 0.45rem;">
                    <span style="color: #ef4444; display: flex; align-items: center; gap: 0.4rem;">
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: #ef4444; display: inline-block;"></span>
                        Negative {neg_pct:.1f}%
                    </span>
                    <span style="color: #10b981; display: flex; align-items: center; gap: 0.4rem;">
                        Positive {pos_pct:.1f}%
                        <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;"></span>
                    </span>
                </div>
                <div style="width: 100%; height: 12px; background: rgba(128, 128, 128, 0.18); border-radius: 9999px; overflow: hidden; display: flex; box-shadow: inset 0 1px 3px rgba(0,0,0,0.15);">
                    <div style="width: {neg_pct}%; background: linear-gradient(90deg, #f43f5e, #ef4444); height: 100%;"></div>
                    <div style="width: {pos_pct}%; background: linear-gradient(90deg, #10b981, #059669); height: 100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Themed Color Telemetry Metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card-indigo">
                    <div class="stat-label-themed">Model Score (Sigmoid)</div>
                    <div class="metric-val-indigo">{score:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card-cyan">
                    <div class="stat-label-themed">Sequence Length</div>
                    <div class="metric-val-cyan">{min(result['total_words'], MAX_LENGTH)} <span style="font-size: 0.82rem; font-weight: 500; opacity: 0.7;">/ {MAX_LENGTH}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card-amber">
                    <div class="stat-label-themed">Total Words</div>
                    <div class="metric-val-amber">{result['total_words']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="metric-card-purple">
                    <div class="stat-label-themed">Out of Vocab Tokens</div>
                    <div class="metric-val-purple">{result['oov_words']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
            <div class="awaiting-card">
                <span class="awaiting-badge">Inference Engine Ready</span>
                <div style="font-weight: 700; font-size: 1.15rem; margin-bottom: 0.35rem;">
                    Awaiting Review Analysis
                </div>
                <div style="font-size: 0.9rem; opacity: 0.75; max-width: 320px; margin: 0 auto; line-height: 1.5;">
                    Select an evaluation sample or type a custom movie review, then click <strong style="color: #6366f1;">Analyze Sentiment</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; font-size: 0.82rem; opacity: 0.7; padding-top: 0.25rem;">
        Sentiment Analysis Engine &bull; LSTM Architecture with Word Embeddings &bull; Built with TensorFlow and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
