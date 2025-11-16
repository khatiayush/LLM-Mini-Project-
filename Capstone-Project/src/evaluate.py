# src/evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import classification_report
import time
import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

# Transformer imports will be optional
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANS_AVAILABLE = True
except ImportError:
    TRANS_AVAILABLE = False


# ---------------------- BOW Evaluation ----------------------
def eval_bow(model_path, vec_path, test_csv):
    print("\n=== Evaluating BOW Model ===")
    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)

    test = pd.read_csv(test_csv)
    X = vec.transform(test["text"])

    t0 = time.time()
    preds = clf.predict(X)
    t1 = time.time()

    print(classification_report(test["label"], preds))
    print(f"Inference time: {t1 - t0:.2f} seconds")


# ---------------------- Word2Vec Evaluation ----------------------
def eval_w2v(clf_path, w2v_path, test_csv):
    if not os.path.exists(clf_path) or not os.path.exists(w2v_path):
        print("\nW2V model not found — skipping.")
        return

    print("\n=== Evaluating Word2Vec Model ===")

    clf = joblib.load(clf_path)
    w2v = KeyedVectors.load(w2v_path).wv

    test = pd.read_csv(test_csv)

    def avg_emb(texts):
        X = []
        dim = w2v.vector_size
        for t in texts:
            tokens = simple_preprocess(str(t))
            vecs = [w2v[w] for w in tokens if w in w2v.key_to_index]
            if len(vecs) == 0:
                X.append(np.zeros(dim))
            else:
                X.append(np.mean(vecs, axis=0))
        return np.vstack(X)

    X_test = avg_emb(test["text"].tolist())
    preds = clf.predict(X_test)

    print(classification_report(test["label"], preds))


# ---------------------- Transformer Evaluation ----------------------
def eval_transformer(model_dir, test_csv):
    if not TRANS_AVAILABLE:
        print("\nTransformers not installed — skipping.")
        return
    if not os.path.exists(model_dir):
        print("\nTransformer model folder not found — skipping.")
        return

    print("\n=== Evaluating Transformer Model ===")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    test = pd.read_csv(test_csv)
    texts = list(test["text"])

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1).cpu().numpy()

    print(classification_report(test["label"], preds))


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--bow_model", default="models/bow.joblib")
    parser.add_argument("--tfidf", default="models/tfidf.joblib")
    parser.add_argument("--w2v_clf", default="models/w2v_clf.joblib")
    parser.add_argument("--w2v_model", default="models/w2v.model")
    parser.add_argument("--transformer", default="models/transformer")

    args = parser.parse_args()

    eval_bow(args.bow_model, args.tfidf, args.test)
    eval_w2v(args.w2v_clf, args.w2v_model, args.test)
    eval_transformer(args.transformer, args.test)
