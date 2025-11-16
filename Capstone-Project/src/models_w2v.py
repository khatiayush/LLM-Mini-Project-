import argparse
import os
import joblib
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


def sentence_tokenize(texts):
    return [simple_preprocess(t) for t in texts]


def build_avg_embeddings(w2v, tokenized):
    dim = w2v.vector_size
    vectors = []
    for tokens in tokenized:
        vecs = [w2v[w] for w in tokens if w in w2v.key_to_index]
        if len(vecs) == 0:
            vectors.append(np.zeros(dim))
        else:
            vectors.append(np.mean(vecs, axis=0))
    return np.vstack(vectors)


def train_w2v(train_path, val_path, model_out='models/w2v_clf.joblib', w2v_out='models/w2v.model'):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    tokenized = sentence_tokenize(train['text'].tolist())

    w2v = Word2Vec(
        sentences=tokenized,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
    w2v.save(w2v_out)

    X_train = build_avg_embeddings(w2v.wv, tokenized)
    y_train = train['label'].values

    tokenized_val = sentence_tokenize(val['text'].tolist())
    X_val = build_avg_embeddings(w2v.wv, tokenized_val)
    y_val = val['label'].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    print(classification_report(y_val, preds))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, model_out)
    print('Saved classifier to', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    args = parser.parse_args()

    train_w2v(args.train, args.val)
