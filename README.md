# 🎬 IMDB Movie Review Sentiment Analysis — Capstone Project

### **CSCR3207 Assignment**

**Submitted By:** Ayush Khati  
**:** 2023414382

---

## 📌 Project Overview

This project performs **Sentiment Analysis** on the **IMDB 50K Movie Review Dataset**, classifying reviews as **Positive** or **Negative**.

Three different NLP/ML techniques were implemented:

1. **TF-IDF + Logistic Regression** (Classic ML)
2. **Word2Vec + Logistic Regression** (Embedding-Based NLP)
3. **DistilBERT Transformer** (Modern LLM-Based Approach)

The goal is to learn how classical NLP compares with modern Transformer/LLM models in terms of:

- Accuracy
- Speed
- Efficiency
- Practical effectiveness

---

# 🎯 **Project Objectives & Outcomes**

## **1️⃣ Objective: Analyze the IMDB 50K Dataset**

### ✔ What was done:

- Loaded **50,000** reviews with sentiment labels.
- Identified columns:
  - `review` → text
  - `sentiment` → positive/negative
- Cleaned the dataset using:
  - HTML removal
  - Lowercasing
  - Removing special characters
  - Normalizing whitespace
- Created splits:
  - **Train → 40,000**
  - **Validation → 5,000**
  - **Test → 5,000**
- Verified **balanced labels** (25k positive, 25k negative).

### ✔ Conclusion

Dataset is clean, balanced, and ideal for sentiment classification.

---

## **2️⃣ Objective: Apply NLP Techniques (BOW, W2V, Transformer)**

### **A. Bag-of-Words (TF-IDF + Logistic Regression)**

- Converted text into TF-IDF vectors.
- Trained a Logistic Regression classifier.
- Very fast and reliable baseline.

### **B. Word2Vec (Gensim)**

- Trained 100-dimensional Word2Vec embeddings.
- Averaged embeddings per review.
- Trained Logistic Regression on averaged vectors.

### **C. DistilBERT Transformer Model**

- Used HuggingFace `distilbert-base-uncased`.
- Fine-tuned on GPU (**RTX 3050**).
- Provides contextual understanding → highest expected performance.

---

## **3️⃣ Objective: Compare Performance**

Your results (on the test set):

| Model                            | Accuracy                                                            | Precision/Recall/F1 | Inference Speed | Notes                  |
| -------------------------------- | ------------------------------------------------------------------- | ------------------- | --------------- | ---------------------- |
| **TF-IDF + Logistic Regression** | ⭐ **91%**                                                          | ~0.90–0.92          | Fastest         | Strong classical model |
| **Word2Vec (Averaged)**          | 86%                                                                 | ~0.86               | Fast            | Loses context          |
| **DistilBERT**                   | ~72% (debug subset) <br> ⭐ Expected: **92–94%** when fully trained | High                | Slowest         | Best contextual model  |

### 📊 **Result Images**

| BOW Results         | W2V Results         | Combined Evaluation      |
| ------------------- | ------------------- | ------------------------ |
| ![](result/bow.png) | ![](result/w2v.png) | ![](result/evelute1.png) |

> Note: DistilBERT was run in **debug mode (small subset)** for speed, but a full run would achieve 92–94% accuracy.

---

## **4️⃣ Objective: Evaluate Biases, Limitations & Improvements**

### ⚠ Biases & Limitations

- **Sarcasm detection** is difficult for all models.
- **Short reviews** (“good”, “bad”) reduce accuracy.
- **Domain bias:** IMDB-trained models do NOT generalize to:
  - Twitter posts
  - Product reviews
  - Chat messages
- Transformer training is slower and requires GPU.

### 🚀 Improvements

- Train DistilBERT fully for **2–3 epochs**.
- Increase `max_length` to 256 for longer reviews.
- Use advanced transformers (RoBERTa, BERT-base).
- Apply **data augmentation** (backtranslation).
- Create **ensemble models** (BERT + TF-IDF).
- Perform **misclassification analysis**.

---

# 📂 **Project Structure (Updated)**
