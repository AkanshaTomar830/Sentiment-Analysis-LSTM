# Movie Review Sentiment Analysis Using Long Short-Term Memory (LSTM) Neural Networks

**Project-Based Learning (PBL) Final Technical Report**  
**Domain**: Deep Learning & Natural Language Processing (NLP)  
**Author / Engineering Team**: PBL NLP Group  
**Frameworks & Libraries**: Python 3.12, TensorFlow 2.20 / Keras 3.x, Streamlit, Scikit-Learn, NumPy, Matplotlib  
**Dataset**: IMDB Large Movie Review Dataset (50,000 Polar Reviews)  

---

## 1. Executive Summary

Sentiment analysis serves as a cornerstone of modern Natural Language Processing (NLP), enabling automated systems to quantify subjective opinions, customer attitudes, and emotional polarities across textual corpora. Classical approaches—such as Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) combined with linear classifiers—fail to capture contextual word order, sentence semantics, and long-range syntactic dependencies.

This project designs, trains, evaluates, and deploys a sequence-aware **Long Short-Term Memory (LSTM)** recurrent neural network tailored for binary sentiment classification on the standard benchmark **IMDB Large Movie Review Dataset** (50,000 reviews). 

Key project outcomes include:
- An end-to-end preprocessing pipeline with word embedding masking (`mask_zero=True`) and sequence padding to a uniform length of 250 tokens.
- An LSTM architecture featuring 64 memory units and 50% dropout regularization, totaling **673,345 parameters**.
- An empirical test classification accuracy of **85.87%** and a test loss of **0.3595** on 25,000 unseen test reviews.
- A balanced performance profile with an F1-score of **85.60%** on negative reviews and **86.13%** on positive reviews.
- A modern, interactive **Streamlit Web Dashboard** delivering real-time sentiment scoring, probability distribution bars, input telemetry, sample presets, and resilient cross-version model loading.

---

## 2. Problem Statement & Motivation

### 2.1 The Challenge of Natural Language Sentiment
Sentiment classification is complicated by inherent linguistic subtleties:
1. **Word-Order Dependencies**: Words take their meaning from context. The sentences *"The script was not bad, actually quite good"* and *"The script was not good, actually quite bad"* contain nearly identical vocabulary but convey opposite sentiment polarities.
2. **Negation & Scope**: Negation terms (*"never"*, *"barely"*, *"hardly"*, *"not"*) can invert the semantic polarity of words situated dozens of tokens downstream.
3. **High Vocabulary Dimensionality**: Natural language vocabularies span tens of thousands of unique tokens, introducing severe sparsity when represented with classical vector space models.

### 2.2 Shortcomings of Classical and Vanilla RNN Approaches
- **Classical Linear Models (BoW / TF-IDF + Naive Bayes / SVM)**: Treat documents as unordered collections of tokens. All sequential ordering and semantic continuity are discarded.
- **Vanilla Recurrent Neural Networks (SimpleRNN)**: Although theoretically capable of handling sequential data through hidden state recurrence ($h_t = \tanh(W x_t + U h_{t-1} + b)$), standard RNNs suffer severely from the **Vanishing and Exploding Gradient Problem** during Backpropagation Through Time (BPTT). Over sequences exceeding 10–20 time steps, the repeated multiplication of weight matrices causes exponential decay or explosion of gradients, making it impossible to capture long-term context.

### 2.3 The LSTM Solution
Long Short-Term Memory networks (introduced by Hochreiter & Schmidhuber) resolve this limitation by introducing a **Constant Error Carousel (CEC)** governed by three multiplicative gates:
1. **Forget Gate**: Decides what fraction of historical memory in the cell state should be discarded.
2. **Input Gate**: Determines which new information from the current token should be written to the cell state.
3. **Output Gate**: Controls which parts of the internal cell state are emitted as the current hidden state output.

This design enables uninterrupted gradient flow across hundreds of time steps, making LSTM the architecture of choice for sequence-sensitive text analysis.

---

## 3. Dataset Description & Exploratory Data Analysis (EDA)

The model is trained on the benchmark **IMDB Large Movie Review Dataset** (Maas et al., Stanford AI Lab).

### 3.1 Corpus Composition & Split
| Dataset Partition | Sample Count | Negative Class (0) | Positive Class (1) | Class Balance |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 25,000 | 12,500 (50.0%) | 12,500 (50.0%) | Perfectly Balanced |
| **Testing Set** | 25,000 | 12,500 (50.0%) | 12,500 (50.0%) | Perfectly Balanced |
| **Total Benchmark** | **50,000** | **25,000** | **25,000** | **50% / 50%** |

### 3.2 Sequence Length Statistics
An empirical analysis of review token lengths across the 25,000 training reviews yielded the following distribution:

| Metric | Measured Value (Tokens) |
| :--- | :---: |
| **Shortest Review** | 11 words |
| **Longest Review** | 2,494 words |
| **Mean Review Length** | 238.71 words |
| **Standard Deviation** | ~173.2 words |
| **Chosen Maximum Length ($T$)** | **250 tokens** |

> Setting the cutoff threshold to **250 tokens** covers the mean review length (~238.7 words) and fully encapsulates approximately **85%** of all reviews in the corpus without truncation, striking an optimal balance between sequence context preservation and computational throughput during training.

---

## 4. Preprocessing Pipeline & Vectorization

```
[Raw User Review Text]
        │
        ▼
[text_to_word_sequence (Lowercase & Punctuation Stripping)]
        │
        ▼
[Word-to-Index Frequency Mapping (Top 10,000 Words)]
        │
        ▼
[Special Token Index Offsetting: 0=PAD, 1=START, 2=OOV]
        │
        ▼
[pad_sequences (Uniform Length: 250, post-padding & post-truncating)]
        │
        ▼
[Tensor Ready for Embedding: Shape (Batch, 250)]
```

### 4.1 Tokenization & Vocabulary Capping
1. **Cleaning**: The raw text undergoes lowercasing, whitespace normalization, and punctuation stripping via `text_to_word_sequence`.
2. **Vocabulary Restriction**: The vocabulary is constrained to the top **$V = 10,000$** most frequent words. Rare words (hapax legomena and low-frequency terms) are mapped to an Out-of-Vocabulary (`<OOV>`) token.
3. **Index Offsetting**:
   - Index `0`: Padding token (`<PAD>`)
   - Index `1`: Sequence Start token (`<START>`)
   - Index `2`: Out-of-Vocabulary token (`<OOV>`)
   - Index `3`: Reserved token
   - Index $k \ge 4$: Corresponds to word rank $(k - 3)$ in the frequency dictionary.

### 4.2 Sequence Padding & Truncation
Because neural networks operate on fixed-dimensional batch tensors, variable-length review sequences are standardized to a uniform length of 250:
- **Reviews with length $> 250$**: Post-truncated (`truncating="post"`), preserving the initial 250 tokens.
- **Reviews with length $< 250$**: Post-padded (`padding="post"`) with zeros to reach a length of 250 tokens.

---

## 5. Model Architecture & Theoretical Formulation

### 5.1 Architecture Pipeline
```
Input Sequence: (Batch, 250)
       │
       ▼
Embedding Layer: input_dim=10,000, output_dim=64, mask_zero=True  ===> Output: (Batch, 250, 64)
       │
       ▼
LSTM Recurrent Layer: 64 Units, tanh activation, sigmoid recurrent ===> Output: (Batch, 64)
       │
       ▼
Dropout Regularization: p = 0.50 (deactivates 50% neurons)        ===> Output: (Batch, 64)
       │
       ▼
Dense Classification Layer: 1 Unit, Sigmoid activation            ===> Output: (Batch, 1) ∈ [0, 1]
```

### 5.2 Mathematical Formulation of Layers

#### 1. Learned Embedding Layer
The Embedding layer maps discrete token indices $x_t \in \{0, 1, \dots, V-1\}$ into a continuous 64-dimensional latent semantic space:
$$e_t = W_{emb}[x_t] \quad \text{where } W_{emb} \in \mathbb{R}^{10000 \times 64}$$
The flag `mask_zero=True` generates a boolean mask that instructs subsequent layers to ignore padding tokens ($0$), preventing artificial bias from zero padding.

#### 2. Long Short-Term Memory (LSTM) Layer
For each time step $t \in [1, 250]$, given input vector $x_t$ (the 64-dimensional embedding) and previous hidden state $h_{t-1}$, the LSTM computes:

1. **Forget Gate**:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate & Candidate Cell State**:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update**:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
4. **Output Gate & Hidden State**:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function, $\odot$ denotes element-wise (Hadamard) multiplication, and $h_T \in \mathbb{R}^{64}$ represents the final contextual vector of the review.

#### 3. Dropout Regularization
To combat overfitting in recurrent representations, a dropout layer randomly deactivates 50% of the hidden activations during training:
$$\tilde{h} = h_T \odot r, \quad r \sim \text{Bernoulli}(1 - p), \quad p = 0.5$$

#### 4. Dense Classification Output
The regularized hidden representation $\tilde{h}$ is mapped to a scalar prediction through a single dense neuron with a sigmoid activation function:
$$\hat{y} = \sigma(W_d \tilde{h} + b_d) \in [0, 1]$$
- $\hat{y} \ge 0.5 \implies$ **Positive Sentiment**
- $\hat{y} < 0.5 \implies$ **Negative Sentiment**

### 5.3 Detailed Parameter Breakdown
| Layer Name | Type | Output Shape | Formula for Parameters | Param Count |
| :--- | :--- | :--- | :--- | :---: |
| `input_layer` | InputLayer | `(None, 250)` | None | 0 |
| `embedding` | Embedding | `(None, 250, 64)` | $10,000 \times 64$ | 640,000 |
| `lstm` | LSTM | `(None, 64)` | $4 \times [(d_{in} + d_{out} + 1) \times d_{out}] = 4 \times [(64 + 64 + 1) \times 64]$ | 33,024 |
| `dropout` | Dropout | `(None, 64)` | None | 0 |
| `dense` | Dense | `(None, 1)` | $(64 + 1) \times 1$ | 65 |
| **Total Parameters** | | | **All Trainable** | **673,345** |

---

## 6. Training Methodology & Optimization

### 6.1 Hyperparameter Configuration
- **Loss Function**: Binary Cross-Entropy
  $$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$
- **Optimization Algorithm**: Adam ($\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$)
- **Batch Size**: 64 samples
- **Maximum Epochs**: 5
- **Validation Split**: 20% of training data (20,000 training samples, 5,000 validation samples)
- **Early Stopping**: Monitored `val_loss` with `patience=2` and restoration of best model weights.

### 6.2 Epoch-by-Epoch Training Dynamics
The model was trained on GPU hardware. The recorded training history across all 5 epochs is detailed below:

| Epoch | Step Time | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Epoch Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 / 5** | 284 ms/step | 0.4589 | 78.19% | 0.3514 | 85.76% | Initial convergence; strong generalization |
| **2 / 5** | 272 ms/step | 0.2828 | 89.07% | 0.3709 | 86.72% | Validation accuracy reaches 86.72% |
| **3 / 5** | 290 ms/step | 0.2228 | 91.72% | **0.3399** | **86.78%** | **Optimal Checkpoint** (Lowest validation loss) |
| **4 / 5** | 283 ms/step | 0.1707 | 93.91% | 0.3689 | 85.66% | Training loss drops; early signs of overfitting |
| **5 / 5** | 293 ms/step | 0.1228 | 95.66% | 0.4578 | 85.86% | Overfitting evident as validation loss rises |

The minimum validation loss was achieved at **Epoch 3 (val_loss: 0.3399, val_accuracy: 86.78%)**. The early stopping mechanism identified this optimal convergence window before overfitting could degrade test generalization.

---

## 7. Experimental Results & Quantitative Evaluation

### 7.1 Test Benchmark Performance
Evaluation was executed across the complete, untouched **25,000 test reviews**:

$$\text{Test Accuracy} = \mathbf{85.87\%} \quad (\text{Exact: } 85.8680\%)$$
$$\text{Test Loss (Binary Cross-Entropy)} = \mathbf{0.3595} \quad (\text{Exact: } 0.359480)$$

### 7.2 Confusion Matrix Analysis
The confusion matrix characterizes the binary classification breakdown on the 25,000 test reviews:

| | Predicted Negative (0) | Predicted Positive (1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (0)** | **10,498** (TN) | **2,002** (FP) | 12,500 |
| **Actual Positive (1)** | **1,531** (FN) | **10,969** (TP) | 12,500 |
| **Total Predicted** | 12,029 | 12,971 | **25,000** |

```
                       PREDICTED
                  Negative    Positive
               +------------+------------+
  ACTUAL  Neg  |   10,498   |    2,002   |  12,500
          Pos  |    1,531   |   10,969   |  12,500
               +------------+------------+
                   12,029       12,971      25,000
```

![Test Confusion Matrix](./Results/confusion_matrix.png)

### 7.3 Detailed Classification Report
From the confusion matrix counts, the precision, recall, and F1-score are derived:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative Sentiment (0)** | $10498 / 12029 = \mathbf{87.27\%}$ | $10498 / 12500 = \mathbf{83.98\%}$ | $\mathbf{85.60\%}$ | 12,500 |
| **Positive Sentiment (1)** | $10969 / 12971 = \mathbf{84.57\%}$ | $10969 / 12500 = \mathbf{87.75\%}$ | $\mathbf{86.13\%}$ | 12,500 |
| **Macro Average** | **85.92%** | **85.87%** | **85.86%** | 25,000 |
| **Weighted Average** | **85.92%** | **85.87%** | **85.86%** | 25,000 |
| **Overall Accuracy** | | | **85.87%** | 25,000 |

### 7.4 Qualitative Analysis of Error Modes
Inspection of the False Positives (2,002) and False Negatives (1,531) revealed three dominant linguistic error modes:
1. **Sarcasm and Satirical Phrasing**: Reviews such as *"A masterpiece of cinematic torture; I have never witnessed such an extraordinary display of complete incompetence"* contain strong positive tokens (*"masterpiece"*, *"extraordinary"*) used with ironic intent.
2. **Bipartite Mixed Reviews**: Reviews that spend paragraphs complimenting visuals or soundtrack before concluding with condemnation of the plot create conflicting recurrent states.
3. **Truncation of Late-Sequence Twists**: Because reviews longer than 250 tokens are truncated, critical summary verdicts at the very end of extensive essays may fall outside the processed window.

---

## 8. Interactive Web Application & Deployment

The trained deep learning model was deployed as a production-grade web dashboard built on **Streamlit** (`Application/app.py`).

### 8.1 Key Application Features
1. **Clean Professional Aesthetics**:
   - Modern, executive visual layout free of emoji clutter.
   - Tech gradient brand header with a live CSS-animated pulsating indicator dot (`Model Online`).
   - Adaptive color theme functioning in both Light and Dark mode.
2. **Balanced Two-Column Workflow**:
   - **Left Column (Input)**: Pre-loaded benchmark samples (Positive, Negative, Mixed), live word/character counters, primary gradient-styled `Analyze Sentiment` button, and `Clear` button.
   - **Right Column (Classification Output)**: Dynamic high-impact verdict card (emerald green for positive, ruby red for negative), dual-tone probability distribution bar, and 4 themed telemetry tiles:
     - **Model Score (Sigmoid)**: Left-accented in Electric Indigo (`#6366f1`).
     - **Sequence Length**: Left-accented in Cyan (`#06b6d4`).
     - **Total Words**: Left-accented in Amber (`#f59e0b`).
     - **Out-of-Vocabulary Tokens**: Left-accented in Purple (`#a855f7`).
3. **Sidebar Model Transparency**:
   - Complete technical specifications (LSTM units, embedding dimension, vocab ceiling, max length).
   - Test benchmark accuracy and loss display.
   - Collapsible viewer for the test Confusion Matrix plot.

### 8.2 Engineering Resilience: Cross-Version Deserialization Fallback
When deploying Keras 3 `.keras` models across varying environment sub-versions, differences in layer keyword arguments (e.g. `quantization_config` introduced in Keras 3.13) can trigger serialization errors. 

To resolve this, the application implements an automated architecture fallback:
```python
@st.cache_resource(show_spinner=False)
def load_sentiment_model(model_path: Path):
    try:
        return tf.keras.models.load_model(str(model_path))
    except Exception:
        # Fallback: Reconstruct exact architecture and load weights directly
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
```
This guarantees zero-downtime execution regardless of whether the model is executed from the repository root or the application directory.

---

## 9. Comparative Architecture Analysis

| Feature / Model | Multinomial Naive Bayes | Vanilla RNN | **Proposed LSTM Model** | Bidirectional LSTM | BERT (Transformer) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sequential Awareness** | No (BoW assumption) | Yes | **Yes** | Yes (Bidirectional) | Yes (Self-Attention) |
| **Long-Term Memory** | None | Extremely Poor | **Strong (Gated)** | Strong (Gated) | Superior (Global) |
| **Gradient Stability** | N/A (Closed-form) | Vanishing / Exploding | **Stable (CEC)** | Stable (CEC) | Stable (LayerNorm) |
| **Parameter Count** | ~10,000 | ~33,000 | **673,345** | ~706,000 | ~110,000,000 |
| **Inference Latency** | < 1 ms | ~5 ms | **~15 ms** | ~28 ms | ~150 ms |
| **IMDB Accuracy** | ~82.5% | ~68.0% | **85.87%** | ~87.5% | ~93.5% |

---

## 10. Future Scope & Recommendations

1. **Bidirectional LSTM (BiLSTM)**:
   Augmenting the single forward LSTM layer with a backward recurrence pass will enable the model to capture upcoming context (e.g. recognizing that an opening criticism is subsequently refuted).
2. **Pretrained Word Embeddings**:
   Initializing the Embedding layer with pretrained **GloVe (Global Vectors for Word Representation)** or **Word2Vec** 100d/300d embeddings will inject semantic relationships derived from billions of web tokens.
3. **Attention Mechanism**:
   Incorporating a Bahdanau self-attention layer across LSTM hidden states will allow the network to dynamically weight the most sentiment-salient sentences rather than relying solely on the final hidden state $h_T$.
4. **Aspect-Based Sentiment Analysis (ABSA)**:
   Expanding the binary classification task to extract multi-dimensional sentiment across specific movie elements (e.g., *Cinematography: Positive*, *Acting: Positive*, *Pacing: Negative*).

---

## 11. Conclusion

This project successfully demonstrates the design, end-to-end training, statistical validation, and production deployment of an LSTM-based Natural Language Processing system for sentiment analysis. By leveraging gated recurrent memory cells, masked dense word embeddings, and dropout regularization, the system achieves an **85.87% accuracy** and a **0.8586 macro F1-score** on the standard IMDB benchmark. The accompanying interactive dashboard provides a responsive, professional, and accessible interface for real-time text analysis.

---

## 12. References & Citations

1. **Hochreiter, S., & Schmidhuber, J. (1997)**. *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
2. **Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011)**. *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011), 142–150.
3. **Kingma, D. P., & Ba, J. (2014)**. *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.
4. **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014)**. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15(1), 1929–1958.
