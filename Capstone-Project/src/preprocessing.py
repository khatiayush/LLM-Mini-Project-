# src/preprocessing.py
import re
from bs4 import BeautifulSoup
import nltk
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import os
import sys

# download once (quiet)
nltk.download('punkt', quiet=True)


def clean_text(text: str) -> str:
    """Lightweight cleaning: strip HTML, lowercase, remove odd characters, normalize spaces."""
    if pd.isna(text):
        return ""
    # remove HTML
    text = BeautifulSoup(str(text), "html.parser").get_text()
    # lower
    text = text.lower()
    # remove non-alphanumeric (keep basic punctuation)
    text = re.sub(r"[^a-z0-9\s\.,!?\'\"]+", " ", text)
    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_columns(df: pd.DataFrame):
    """Try to find text and label columns from common names; otherwise assume first two cols."""
    text_cols = ['review', 'text', 'content', 'movie_review']
    label_cols = ['sentiment', 'label', 'target', 'rating']

    text_col = None
    label_col = None

    for col in df.columns:
        if col.lower() in text_cols:
            text_col = col
        if col.lower() in label_cols:
            label_col = col

    if text_col is None:
        # fallback: first column
        text_col = df.columns[0]
    if label_col is None:
        # fallback: second column if exists
        if len(df.columns) > 1:
            label_col = df.columns[1]
        else:
            raise ValueError("Couldn't detect a label column. Provide a CSV with at least two columns.")

    return text_col, label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    """Convert common label formats to 0/1 integers."""
    # If already numeric 0/1
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        # round and clip to 0/1
        s = series.fillna(0).astype(int)
        s = s.clip(0, 1)
        return s

    s = series.astype(str).str.lower().str.strip()

    mapping = {
        'positive': 1, 'pos': 1, '1': 1, 'true': 1, 't': 1, 'y': 1, 'yes': 1,
        'negative': 0, 'neg': 0, '0': 0, 'false': 0, 'f': 0, 'n': 0, 'no': 0
    }

    # map known words
    s_mapped = s.map(lambda x: mapping.get(x, None))

    # If still None, try numeric strings
    if s_mapped.isnull().any():
        def try_int(x):
            try:
                xi = int(float(x))
                return 1 if xi >= 1 else 0
            except Exception:
                return None
        s_mapped = s_mapped.fillna(s.map(try_int))

    if s_mapped.isnull().any():
        # If there are still unmapped values, raise informative error
        unknown = pd.Series(s[s_mapped.isnull()].unique())
        raise ValueError(f"Found unknown label values: {unknown.tolist()}. "
                         "Please convert labels to positive/negative or 1/0.")

    return s_mapped.astype(int)


def load_and_split(path: str, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42, verbose: bool = True):
    # Read with fallback encoding handling
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except Exception:
        df = pd.read_csv(path, encoding='latin1', engine='python')

    if verbose:
        print(f"Loaded CSV with shape: {df.shape}")

    text_col, label_col = detect_columns(df)
    if verbose:
        print(f"Detected text column: '{text_col}'  label column: '{label_col}'")

    texts = df[text_col]
    labels = df[label_col]

    labels = normalize_labels(labels)

    texts = texts.fillna("").apply(clean_text)

    # stratify requires no missing labels and balanced enough samples
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=random_state, stratify=y_train_val
    )

    train = pd.DataFrame({'text': X_train, 'label': y_train})
    val = pd.DataFrame({'text': X_val, 'label': y_val})
    test = pd.DataFrame({'text': X_test, 'label': y_test})

    if verbose:
        print("Split sizes -> train: {}, val: {}, test: {}".format(len(train), len(val), len(test)))
        print("Label distribution (train):")
        print(train['label'].value_counts().to_dict())

    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, out_dir: str = "data/splits", verbose: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    if verbose:
        print(f"Saved train/val/test into '{out_dir}'")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess IMDB CSV and create train/val/test splits.")
    parser.add_argument("--input", type=str, required=True, help="Path to IMDB CSV file")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Directory to save splits")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion for test set")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion for validation set (of the full dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress print output")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    verbose = not args.quiet
    try:
        train_df, val_df, test_df = load_and_split(
            args.input, test_size=args.test_size, val_size=args.val_size, random_state=args.seed, verbose=verbose
        )
        save_splits(train_df, val_df, test_df, out_dir=args.out_dir, verbose=verbose)

        if verbose:
            # show a few examples
            print("\nSample training rows:")
            print(train_df.head(3).to_string(index=False))
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
