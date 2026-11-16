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
# Professional Styling (Light & Dark Mode Adaptive)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Clean layout & typography */
    .stTextArea textarea {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 0.95rem;
        line-height: 1.5;
        border-radius: 8px;
    }
    
    /* Subtle container cards */
    .metric-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    
    .verdict-positive {
        background-color: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .verdict-negative {
        background-color: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .status-badge-pos {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #059669;
        background: rgba(16, 185, 129, 0.16);
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        margin-bottom: 0.45rem;
    }
    
    .status-badge-neg {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #dc2626;
        background: rgba(239, 68, 68, 0.16);
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        margin-bottom: 0.45rem;
    }
    
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin-bottom: 0.6rem;
    }
    
    .stat-label {
        font-size: 0.76rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .awaiting-card {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px dashed rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 2.75rem 1.5rem;
        text-align: center;
    }

    .awaiting-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }

    .awaiting-desc {
        font-size: 0.88rem;
        opacity: 0.7;
        max-width: 320px;
        margin: 0 auto;
        line-height: 1.5;
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
        **Architecture**: Recurrent Neural Network (LSTM)  
        **Task**: Binary Sentiment Classification  
        **Corpus**: IMDB 50,000 Movie Reviews  
        """
    )
    
    st.divider()
    
    st.markdown("#### Hyperparameters")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric(label="Vocab Limit", value=f"{VOCAB_SIZE:,}")
        st.metric(label="Embedding Dim", value="64")
    with col_sb2:
        st.metric(label="Max Sequence", value=f"{MAX_LENGTH} tokens")
        st.metric(label="LSTM Units", value="64")

    st.divider()

    st.markdown("#### Test Set Performance")
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
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 1.5rem; border-bottom: 1px solid rgba(128, 128, 128, 0.2); padding-bottom: 0.85rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <h2 style="margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em;">Movie Review Sentiment Analysis</h2>
                <p style="margin: 0.25rem 0 0 0; opacity: 0.75; font-size: 0.92rem;">
                    Natural language classification powered by a sequence-aware LSTM recurrent neural network.
                </p>
            </div>
            <div style="display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.75rem; background: rgba(59, 130, 246, 0.08); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 9999px; font-size: 0.8rem; font-weight: 600; color: #2563eb;">
                <span style="display: inline-block; width: 7px; height: 7px; background: #2563eb; border-radius: 50%;"></span>
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
    st.markdown('<div class="section-title">Review Input</div>', unsafe_allow_html=True)

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
    st.caption(f"Length: {words_entered} words | {chars_entered} characters")

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
    st.markdown('<div class="section-title">Classification Output</div>', unsafe_allow_html=True)

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
                    <span class="status-badge-pos">Classification</span>
                    <h3 style="margin: 0 0 0.35rem 0; color: #10b981; font-size: 1.45rem; font-weight: 700;">
                        Positive Sentiment
                    </h3>
                    <div style="font-size: 0.9rem; opacity: 0.85;">
                        {confidence_tier} &bull; <strong>{confidence:.1f}%</strong> positive probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="verdict-negative">
                    <span class="status-badge-neg">Classification</span>
                    <h3 style="margin: 0 0 0.35rem 0; color: #ef4444; font-size: 1.45rem; font-weight: 700;">
                        Negative Sentiment
                    </h3>
                    <div style="font-size: 0.9rem; opacity: 0.85;">
                        {confidence_tier} &bull; <strong>{confidence:.1f}%</strong> negative probability
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Probability distribution bar
        st.markdown(
            f"""
            <div style="margin-bottom: 1.25rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem;">
                    <span style="color: #ef4444;">Negative {neg_pct:.1f}%</span>
                    <span style="color: #10b981;">Positive {pos_pct:.1f}%</span>
                </div>
                <div style="width: 100%; height: 9px; background-color: rgba(128, 128, 128, 0.15); border-radius: 9999px; overflow: hidden; display: flex;">
                    <div style="width: {neg_pct}%; background-color: #ef4444; height: 100%;"></div>
                    <div style="width: {pos_pct}%; background-color: #10b981; height: 100%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Telemetry metrics
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="stat-label">Model Score (Sigmoid)</div>
                    <div class="stat-value">{score:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="stat-label">Processed Sequence</div>
                    <div class="stat-value">{min(result['total_words'], MAX_LENGTH)} <span style="font-size: 0.82rem; font-weight: 400; opacity: 0.65;">/ {MAX_LENGTH} max</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="stat-label">Word Count</div>
                    <div class="stat-value">{result['total_words']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="stat-label">Out of Vocab Words</div>
                    <div class="stat-value">{result['oov_words']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.markdown(
            """
            <div class="awaiting-card">
                <div class="awaiting-title">
                    Awaiting Review Analysis
                </div>
                <div class="awaiting-desc">
                    Select an evaluation sample or type a custom movie review, then click <strong>Analyze Sentiment</strong> to generate predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Sentiment Analysis Engine &bull; LSTM Architecture with Word Embeddings &bull; Built with TensorFlow and Streamlit"
)
