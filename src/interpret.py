"""
interpret.py — Giai thich du doan cua mo hinh ABSA.

Hai muc do:

1. GLOBAL  — n-gram nao day mot du doan ve phia moi polarity tren toan bo
             mo hinh. Doc truc tiep tu ma tran he so cua Logistic Regression,
             khong can xap xi.

2. LOCAL   — voi mot cau cu the, dong gop cua tung feature xuat hien trong
             cau do vao diem so cua lop duoc du doan:
                 contribution_j = x_j * w_{c,j}
             Tong cac contribution cong voi intercept bang dung logit cua lop,
             nen day la phan ra CHINH XAC chu khong phai uoc luong nhu LIME
             hay Kernel SHAP.

Voi mo hinh transformer o src/train_transformer.py, dung Integrated Gradients
(Captum) thay cho ham nay; giao dien ham `explain_text` duoc giu nguyen de
hai duong deu goi duoc tu cung mot script danh gia.

Cach dung:
    python src/interpret.py --global-top 15
    python src/interpret.py --text "pin tut nhanh qua, chup anh thi dep" --aspect BATTERY
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.sparse import hstack


def _feature_names(vecs):
    word_vec, char_vec, asp_enc = vecs
    return np.concatenate([
        np.array([f"w:{t}" for t in word_vec.get_feature_names_out()]),
        np.array([f"c:{t}" for t in char_vec.get_feature_names_out()]),
        np.array([f"a:{t}" for t in asp_enc.get_feature_names_out()]),
    ])


def global_importance(bundle, top_k=15, word_only=True):
    """N-gram co trong so lon nhat cho tung polarity.

    word_only=True loc bo char n-gram vi chung kho doc voi nguoi danh gia;
    dat False neu muon kiem tra xem mo hinh co bam vao mau ky tu la khong.
    """
    clf, vecs = bundle["clf"], bundle["vecs"]
    names = _feature_names(vecs)
    keep = np.array([n.startswith("w:") for n in names]) if word_only \
        else np.ones(len(names), dtype=bool)

    out = {}
    for i, cls in enumerate(clf.classes_):
        w = clf.coef_[i].copy()
        w[~keep] = 0.0
        top = np.argsort(w)[::-1][:top_k]
        out[str(cls)] = [
            {"feature": names[j].split(":", 1)[1], "weight": float(w[j])}
            for j in top
        ]
    return out


def explain_text(bundle, text, aspect, top_k=12):
    """Phan ra dong gop cua tung feature cho mot cap (cau, aspect)."""
    clf, vecs = bundle["clf"], bundle["vecs"]
    word_vec, char_vec, asp_enc = vecs

    X = hstack([
        word_vec.transform([text]),
        char_vec.transform([text]),
        asp_enc.transform(pd.DataFrame({"aspect": [aspect]})),
    ]).tocsr()

    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    pred_cls = str(clf.classes_[pred_idx])

    names = _feature_names(vecs)
    w = clf.coef_[pred_idx]
    idx = X.indices
    contrib = X.data * w[idx]

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    detail = [
        {
            "feature": names[idx[k]],
            "value": float(X.data[k]),
            "weight": float(w[idx[k]]),
            "contribution": float(contrib[k]),
        }
        for k in order
    ]

    return {
        "text": text,
        "aspect": aspect,
        "prediction": pred_cls,
        "probabilities": {str(c): float(p) for c, p in zip(clf.classes_, proba)},
        "intercept": float(clf.intercept_[pred_idx]),
        "sum_contributions": float(contrib.sum()),
        "top_features": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_baseline.joblib")
    ap.add_argument("--out", default="results/interpretability.json")
    ap.add_argument("--global-top", type=int, default=15)
    ap.add_argument("--text", default=None)
    ap.add_argument("--aspect", default="GENERAL")
    args = ap.parse_args()

    # Console Windows mac dinh dung cp1252, khong in duoc tieng Viet co dau ->
    # UnicodeEncodeError truoc khi kip ghi file ket qua. Ep stdout ve UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bundle = load(args.model)

    if args.text:
        exp = explain_text(bundle, args.text, args.aspect)
        print(f"\nCau    : {exp['text']}")
        print(f"Aspect : {exp['aspect']}")
        print(f"Du doan: {exp['prediction']}  "
              f"(p={exp['probabilities'][exp['prediction']]:.3f})\n")
        print(f"{'feature':<28}{'weight':>10}{'contrib':>10}")
        print("-" * 48)
        for d in exp["top_features"]:
            print(f"{d['feature']:<28}{d['weight']:>10.3f}"
                  f"{d['contribution']:>10.4f}")
        return

    gi = global_importance(bundle, top_k=args.global_top)
    for cls, items in gi.items():
        print(f"\n--- {cls} ---")
        print(", ".join(f"{d['feature']} ({d['weight']:.2f})" for d in items))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gi, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nDa luu -> {args.out}")


if __name__ == "__main__":
    main()
