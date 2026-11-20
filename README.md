# Movie Review Sentiment Analysis Using LSTM Neural Networks

An end-to-end Natural Language Processing (NLP) and Deep Learning project that classifies movie review sentiment (Positive / Negative) using a sequence-aware **Long Short-Term Memory (LSTM)** neural network. Includes an interactive **Streamlit web application** for real-time sentiment scoring.

---

## Project Overview

* **Domain:** Natural Language Processing (NLP) & Deep Learning
* **Model Architecture:** Embedding Layer + Masking + LSTM (64 units) + Dropout (0.5) + Dense Sigmoid
* **Dataset:** IMDB Large Movie Review Dataset (50,000 Polar Reviews)
* **Test Accuracy:** **85.87%** (Test Loss: 0.3595)
* **F1-Score:** 85.60% (Negative) / 86.13% (Positive)
* **Application:** Interactive Streamlit Dashboard

For complete technical documentation, architecture diagrams, and mathematical formulation, refer to [Project_Report.md](Project_Report.md).

---

## Repository Structure

```text
├── Application/
│   ├── app.py                # Streamlit Web Application
│   └── requirements.txt      # Python dependencies
├── Models/
│   ├── sentiment_lstm_model.keras  # Trained LSTM Model
│   └── word_index.json       # Vocabulary index dictionary
├── Notebooks/
│   └── 01_LSTM_Sentiment_Analysis.ipynb  # End-to-end training notebook
├── Results/
│   └── confusion_matrix.png  # Evaluation confusion matrix
├── Project_Report.md         # Comprehensive PBL Technical Report
├── README.md                 # Project Overview & Setup Guide
└── .gitignore                # Git ignore rules
```

---

## Key Model Highlights

1. **Sequence-Aware Learning:** Unlike traditional Bag-of-Words or TF-IDF representations, the LSTM captures word order, sentence semantics, and long-range contextual dependencies.
2. **Padding & Masking:** Sequences are padded to 250 tokens with zero-masking (`mask_zero=True`) so variable-length reviews are handled without padding bias.
3. **Regularization:** 50% Dropout regularization prevents overfitting across the 673,345 total parameters.

---

## Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AkanshaTomar830/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r Application/requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run Application/app.py
```
Open `http://localhost:8501` in your browser to test real-time sentiment classification.

---

## Author

* **Akansha Tomar** ([GitHub Profile](https://github.com/AkanshaTomar830))
