"""
train_baseline.py — Baseline Aspect Category Sentiment Analysis cho UIT-ViSFD.

Mo hinh: TF-IDF (word 1-2 gram + char 2-5 gram) ket hop one-hot cua aspect,
phan loai bang Logistic Regression (multinomial, class_weight='balanced').

Day la BASELINE co chu dich: chay duoc tren CPU, khong can tai model tu
internet, va he so cua mo hinh tuyen tinh doc duoc truc tiep -> phuc vu
yeu cau interpretability ma khong can post-hoc explainer.

Mo hinh chinh cua luan van (PhoBERT-v2 / ViSoBERT fine-tune, hoac LLM sinh
tuple voi LoRA/DoRA) nam o src/train_transformer.py va can GPU.

Output:
  results/model_baseline.joblib
  results/predictions_test.csv
  results/metrics_overall.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder

SEED = 42


def load(path):
    return pd.read_json(path, lines=True)


def build_features(train, other_splits, max_word=60000, max_char=80000):
    """Khop vectorizer tren TAP TRAIN, sau do transform cac tap con lai.

    Khop rieng tren train la bat buoc: khop tren toan bo du lieu se ro ri
    thong tin tu tap test vao qua trinh huan luyen.
    """
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2,
        max_features=max_word, sublinear_tf=True, lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3,
        max_features=max_char, sublinear_tf=True, lowercase=True,
    )
    asp_enc = OneHotEncoder(handle_unknown="ignore")

    Xw = word_vec.fit_transform(train["comment"])
    Xc = char_vec.fit_transform(train["comment"])
    Xa = asp_enc.fit_transform(train[["aspect"]])
    X_train = hstack([Xw, Xc, Xa]).tocsr()

    outs = []
    for df in other_splits:
        outs.append(hstack([
            word_vec.transform(df["comment"]),
            char_vec.transform(df["comment"]),
            asp_enc.transform(df[["aspect"]]),
        ]).tocsr())

    return X_train, outs, (word_vec, char_vec, asp_enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--C", type=float, default=4.0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load(data_dir / "train.jsonl")
    dev = load(data_dir / "dev.jsonl")
    test = load(data_dir / "test.jsonl")
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    X_train, (X_dev, X_test), vecs = build_features(train, [dev, test])
    y_train = train["polarity"].values

    # lbfgs + multinomial: mot ma tran he so (3 lop x n_features), moi hang
    # doc duoc truc tiep nhu muc do dong gop cua tung n-gram vao mot polarity.
    clf = LogisticRegression(
        C=args.C, max_iter=2000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    clf.fit(X_train, y_train)

    dev_pred = clf.predict(X_dev)
    test_pred = clf.predict(X_test)

    dev_f1 = f1_score(dev["polarity"], dev_pred, average="macro")
    test_f1 = f1_score(test["polarity"], test_pred, average="macro")
    print(f"\nDev  macro-F1 : {dev_f1:.4f}")
    print(f"Test macro-F1 : {test_f1:.4f}\n")
    print(classification_report(test["polarity"], test_pred, digits=4))

    pred_df = test.copy()
    pred_df["pred"] = test_pred
    pred_df.to_csv(out_dir / "predictions_test.csv", index=False)

    import sklearn, scipy, sys
    metrics = {
        "model": "tfidf_logreg_baseline",
        "seed": SEED,
        "C": args.C,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "n_train_pairs": int(len(train)),
        "n_test_pairs": int(len(test)),
        "dev_macro_f1": float(dev_f1),
        "test_macro_f1": float(test_f1),
        "test_report": classification_report(
            test["polarity"], test_pred, output_dict=True, digits=4
        ),
    }
    (out_dir / "metrics_overall.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dump({"clf": clf, "vecs": vecs}, out_dir / "model_baseline.joblib")
    print(f"\nDa luu du doan -> {out_dir/'predictions_test.csv'}")


if __name__ == "__main__":
    main()
