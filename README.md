```markdown
# 🧠 Part-of-Speech Tagging using Hidden Markov Model (HMM)

This mini-project demonstrates how a **Hidden Markov Model (HMM)** can be used for **Part-of-Speech (POS) tagging** — assigning grammatical labels like *noun*, *verb*, *adjective*, etc. to words in a sentence.

---

## 📘 Project Overview

The goal is to implement a simple **HMM-based POS Tagger** that:
- Learns transition and emission probabilities from a tagged corpus  
- Uses the **Viterbi algorithm** to find the most likely POS sequence  
- Visualizes the resulting tag sequence  

This is a **foundational NLP experiment** often used to introduce sequence models before exploring RNNs or Transformers.

---

## 🧩 Architecture / Flow

```

+-------------------------+
|     Input Sentence      |
+-----------+-------------+
|
v
+-------------------------+
| Text Preprocessing      |
| (tokenization, cleanup) |
+-----------+-------------+
|
v
+-------------------------+
| Training HMM Model      |
| (transition + emission) |
+-----------+-------------+
|
v
+-------------------------+
| Apply Viterbi Algorithm |
| (find best tag sequence)|
+-----------+-------------+
|
v
+-------------------------+
| Output POS Tagged Text  |
+-------------------------+

````

---

## 🚀 Features

- Implementation of **Hidden Markov Model** for POS tagging  
- **Viterbi algorithm** for optimal tag decoding  
- Training using **NLTK’s Treebank corpus**  
- Graphical visualization of tag transitions using Matplotlib  
- Simple, self-contained Python implementation  

---

## ⚙️ Requirements

Make sure you have Python ≥ 3.8 installed.  
Install dependencies using:

```bash
pip install nltk matplotlib
````

---

## 🧠 How to Run

### ▶️ Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the Python file or paste the code into a new notebook
3. Run all cells
4. View the POS-tagged output and the plot

### ▶️ Option 2: Local Machine

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/HMM_POS_Tagging.git
   cd HMM_POS_Tagging
   ```

2. Run the script:

   ```bash
   python hmm_pos_tagging.py
   ```

---

## 🧩 Example Output

**Input Sentence:**

```
Time flies like an arrow
```

**Tagged Output:**

```
[('Time', 'NOUN'), ('flies', 'VERB'), ('like', 'ADP'), ('an', 'DET'), ('arrow', 'NOUN')]
```

**Visualization:**
A Matplotlib plot showing POS tag transitions for the given sentence.

---

## 🧰 Tools & Libraries Used

| Tool                       | Purpose                                  |
| -------------------------- | ---------------------------------------- |
| **Python (3.8+)**          | Programming language                     |
| **NLTK**                   | Corpus, tokenization, POS tagging data   |
| **Matplotlib**             | Visualization of POS tags                |
| **Google Colab / Jupyter** | Running and testing the project          |
| **ChatGPT (GPT-5)**        | Documentation and explanation assistance |

---

## 📚 References

1. Jurafsky, D. & Martin, J. H. (2024). *Speech and Language Processing (3rd ed.)*.
   [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
2. GeeksforGeeks (2025). *Part of Speech Tagging with Hidden Markov Model*.
   [https://www.geeksforgeeks.org/nlp/part-of-speech-pos-tagging-with-hidden-markov-model](https://www.geeksforgeeks.org/nlp/part-of-speech-pos-tagging-with-hidden-markov-model)
3. Zhou, C. (2025). *Mastering NLP: Hidden Markov Models for POS Tagging*.
   [https://medium.com/@conniezhou678/mastering-natural-language-processing-part-25-hidden-markov-models-for-pos-tagging-in-nlp-b78891fcff80](https://medium.com/@conniezhou678/mastering-natural-language-processing-part-25-hidden-markov-models-for-pos-tagging-in-nlp-b78891fcff80)
4. Baeldung (2024). *Part-of-Speech Tagging With Hidden Markov Model*.
   [https://www.baeldung.com/cs/nlp-hmm-pos-tags](https://www.baeldung.com/cs/nlp-hmm-pos-tags)

---

## 📸 Results & Screenshots

| Description               | Screenshot                                            |
| ------------------------- | ----------------------------------------------------- |
| Input Sentence            | *(Insert screenshot of input cell)*                   |
| Intermediate Computations | *(Insert transition/emission matrix or print output)* |
| Final Output              | *(Insert tagged result and Matplotlib plot)*          |

---

## 🏁 Learning & Outcomes

* Learned how probabilistic models capture linguistic structure
* Implemented a working HMM from scratch
* Understood the **Viterbi decoding process**
* Gained insight into the differences between classical and deep-learning NLP models

---

## 🧾 License

This project is released under the **MIT License**.
Feel free to fork and modify for educational purposes.

---

**Developed by:**
*Aarush Saboo (Roll No. 16010422097)*
Batch: A, Division 2

```
```
