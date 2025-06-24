# 🔁 RNN, LSTM, and Character-Level Language Modeling

This repository includes:
- NumPy-based implementation of RNN and LSTM (forward & backward passes)
- Character-level language generation (trained on dinosaur names)
- Sampling and optimization loops for text generation

---

## 🧠 Architecture Overview

### 🔷 RNN
- Hidden state propagation using tanh
- Forward and backward pass implemented manually
- Gradient clipping and parameter updates

### 🔶 LSTM
- Four gates: Forget, Input, Candidate, Output
- More stable gradient flow through long sequences
- Full Backpropagation Through Time (BPTT) supported

---

## 📚 Character-Level Language Modeling

- Reads input text (e.g. `dinos.txt`)
- Creates character vocabulary and mappings
- Trains RNN to predict next character
- Generates new sequences by sampling from softmax distribution

---
## ⚙️ Key Components

### 🔸 Forward Pass
- `rnn_cell_forward()` – Forward pass for one RNN cell
- `rnn_forward()` – Loops through sequence for RNN
- `lstm_cell_forward()` – One-step forward pass for LSTM
- `lstm_forward()` – Full LSTM forward pass across sequence

### 🔸 Backward Pass
- `rnn_cell_backward()` – Computes gradients for one RNN step
- `rnn_backward()` – Aggregates gradients for RNN
- `lstm_cell_backward()` – Computes gate derivatives for one step of LSTM
- `lstm_backward()` – Full BPTT over LSTM

### 🧪 Utilities
- `clip()` – Gradient clipping to avoid exploding gradients
- `sample()` – Random sampling from softmax probabilities
- `optimize()` – Forward, backward and update steps in training
- `model()` – Training loop for generating character-level names

---

## 📂 File Structure

```bash
├── rnn_utils.py         # Activation functions, initializations
├── rnn_lstm_main.py     # Full RNN & LSTM logic
├── dinos.txt            # Input data for language modeling
├── utils.py             # Sampling, optimization, gradient clipping
```

---

## 🔬 Sample Outputs

```text
a[4][3][6] = 0.2197
c[1][2][1] = -0.2219
gradients["dWax"].shape = (5,3)
gradients["dWc"].shape = (5,8)
list of sampled characters: ['t', 'o', 'r', 'a', 'n', 'o', 'z', 'a', 'u', 'r']
```

---

## 🔁 Training Loop Example

```python
parameters = model(data, ix_to_char, char_to_ix, num_iterations=10000)
```

- Generates new dinosaur names every 2000 iterations
- Smooths loss with exponential moving average
- Resets and shuffles input sequences for each epoch

---
