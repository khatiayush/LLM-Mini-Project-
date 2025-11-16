import argparse
import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def train_bow(train_path, val_path, model_out='models/bow.joblib', vec_out='models/tfidf.joblib'):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vec.fit_transform(train['text'])
    y_train = train['label']

    X_val = vec.transform(val['text'])
    y_val = val['label']

    clf = LogisticRegression(max_iter=1000)

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = clf.predict(X_val)
    report = classification_report(y_val, preds, digits=4)
    cm = confusion_matrix(y_val, preds)

    print('Validation classification report:\n', report)
    print('Confusion matrix:\n', cm)
    print(f'Training time: {train_time:.2f}s')

    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, model_out)
    joblib.dump(vec, vec_out)
    print('Saved model to', model_out, 'and vectorizer to', vec_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    args = parser.parse_args()

    train_bow(args.train, args.val)
