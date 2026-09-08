"""
sensitivity_threshold.py — Kiem chung phuong an bi loai o muc 6.2 cua bao cao.

Muc 6.2 cua docs/ETHICS.md khang dinh rang co the thu hep khoang cach cong bang
o aspect PRICE bang cach ha nguong quyet dinh rieng cho aspect nay, nhung lam
vay se tang false positive dung o aspect dang yeu va che di nguyen nhan goc.
Cho den truoc script nay, khang dinh do CHUA he duoc chay thu. Mot bao cao dao
duc loai bo mot phuong an ma khong do thu cai gia cua no thi lap luan van con
la suy doan.

Script quet nguong quyet dinh cho lop Negative RIENG o aspect PRICE:

    neu aspect == PRICE:
        pred = Negative                     khi P(Negative) >= t
        pred = argmax cua hai lop con lai   nguoc lai
    nguoc lai:
        pred = argmax nhu mo hinh goc

Voi moi nguong t, script bao cao:
  - TPR, false positive rate va precision cua lop Negative RIENG o PRICE
  - Equal opportunity gap toan cuc theo chieu aspect, kem hai chieu phu
  - Macro-F1 toan cuc tren ca 6.722 cap (comment, aspect) cua tap test

KHONG sua mo hinh chinh. Script chi doc results/model_baseline.joblib va ap mot
quy tac quyet dinh khac len xac suat da co, nen khong the anh huong nguoc lai
bat ky con so nao trong pipeline chinh.

Cach dung:
    python src/sensitivity_threshold.py
    python src/sensitivity_threshold.py --aspect PRICE --step 0.01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from scipy.sparse import hstack
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fairness_check import fairness_report, add_derived_groups  # noqa: E402

POSITIVE = "Negative"


def build_test_matrix(bundle, test):
    """Dung lai ma tran dac trung tap test tu vectorizer DA KHOP tren train.

    Khong khop lai bat ky vectorizer nao, chi transform - dung nhu trong
    train_baseline.py. Nho vay khong co ro ri du lieu, va ket qua trung khop
    voi predictions_test.csv khi quy tac quyet dinh la argmax.
    """
    word_vec, char_vec, asp_enc = bundle["vecs"]
    return hstack([
        word_vec.transform(test["comment"]),
        char_vec.transform(test["comment"]),
        asp_enc.transform(test[["aspect"]]),
    ]).tocsr()


def apply_threshold(proba, classes, base_pred, target_mask, threshold):
    """Ap nguong rieng cho lop POSITIVE tren cac dong thuoc target_mask."""
    pos_i = list(classes).index(POSITIVE)
    other_i = [i for i in range(len(classes)) if i != pos_i]

    pred = base_pred.copy()
    p_pos = proba[:, pos_i]
    # Trong nhom muc tieu: goi Negative khi vuot nguong, nguoc lai chon lop tot
    # nhat trong hai lop con lai (khong bao gio de trong du doan).
    fallback = np.array(classes)[np.array(other_i)][
        np.argmax(proba[:, other_i], axis=1)
    ]
    pred[target_mask] = np.where(
        p_pos[target_mask] >= threshold, POSITIVE, fallback[target_mask]
    )
    return pred


def group_rates(y_true, y_pred):
    """TPR, FPR va precision cua lop POSITIVE tren mot tap con."""
    is_pos = y_true == POSITIVE
    said_pos = y_pred == POSITIVE
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())
    tp = int((is_pos & said_pos).sum())
    fp = int((~is_pos & said_pos).sum())
    return {
        "n": int(len(y_true)),
        "n_true_positive_class": n_pos,
        "tpr": float(tp / n_pos) if n_pos else None,
        "fpr": float(fp / n_neg) if n_neg else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "n_predicted_positive_class": int(said_pos.sum()),
        "n_true_pos_caught": tp,
        "n_false_positive": fp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_baseline.joblib")
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--out", default="results/sensitivity_threshold.json")
    ap.add_argument("--aspect", default="PRICE",
                    help="aspect duoc ha nguong rieng")
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--n-bootstrap", type=int, default=0,
                    help="bootstrap trong moi buoc quet; 0 de tat cho nhanh")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bundle = load(args.model)
    clf = bundle["clf"]
    test = pd.read_json(args.data, lines=True)

    X = build_test_matrix(bundle, test)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    base_pred = clf.predict(X)

    y_true = test["polarity"].to_numpy()
    target_mask = (test["aspect"] == args.aspect).to_numpy()
    if not target_mask.any():
        raise SystemExit(f"Khong co dong nao co aspect = {args.aspect}")

    base_macro_f1 = float(f1_score(y_true, base_pred, average="macro"))
    base_rates = group_rates(y_true[target_mask], base_pred[target_mask])

    thresholds = [float(t) for t in np.round(np.arange(args.step, 1.0, args.step), 4)]
    rows = []
    for t in thresholds:
        pred = apply_threshold(proba, classes, base_pred, target_mask, t)
        rates = group_rates(y_true[target_mask], pred[target_mask])

        df = test[["aspect", "n_star", "n_words"]].copy()
        df["polarity"] = y_true
        df["pred"] = pred
        df = add_derived_groups(df)

        gaps, worst = {}, {}
        for col in ("aspect", "star_bucket", "length_bucket"):
            rep = fairness_report(df, col, positive_class=POSITIVE,
                                  n_boot=args.n_bootstrap)
            gaps[col] = rep["equal_opportunity_gap"]
            worst[col] = rep["worst_group"]

        rows.append({
            "threshold": t,
            **{f"price_{k}": v for k, v in rates.items()},
            "eo_gap_aspect": gaps["aspect"],
            # Nhom yeu nhat theo chieu aspect SAU khi ha nguong. Cot nay la
            # cot dang doc nhat cua ca bang: khi PRICE khong con la nhom yeu
            # nhat, gap khong bien mat ma chuyen sang aspect ke tiep.
            "worst_aspect": (worst["aspect"] or {}).get("group"),
            "worst_aspect_tpr": (worst["aspect"] or {}).get("tpr"),
            "eo_gap_star_bucket": gaps["star_bucket"],
            "eo_gap_length_bucket": gaps["length_bucket"],
            "macro_f1_overall": float(f1_score(y_true, pred, average="macro")),
        })

    # Nguong dau tien dua EO gap theo aspect xuong duoi 0,10 - tuc la muc ma
    # trong mot bao cao se "trong nhu da giai quyet xong".
    closing = next((r for r in rows if r["eo_gap_aspect"] is not None
                    and r["eo_gap_aspect"] < 0.10), None)

    # Ket qua dang chu y nhat cua phep quet: gap KHONG giam vo han theo nguong.
    # Khi PRICE da duoc keo len tren muc cua aspect yeu thu hai, gap bi chan
    # duoi boi chinh aspect do, va viec ha nguong tiep chi con lam hong PRICE.
    scored = [r for r in rows if r["eo_gap_aspect"] is not None]
    floor = min(scored, key=lambda r: r["eo_gap_aspect"]) if scored else None

    out = {
        "purpose": ("Kiem chung dinh luong cho phuong an bi loai o muc 6.2 cua "
                    "docs/ETHICS.md: ha nguong quyet dinh rieng cho mot aspect."),
        "model": args.model,
        "target_aspect": args.aspect,
        "positive_class": POSITIVE,
        "n_test_pairs": int(len(test)),
        "note_no_model_change": ("Script chi doc mo hinh da huan luyen va doi "
                                 "quy tac quyet dinh o khau suy luan. Khong "
                                 "huan luyen lai, khong ghi de model."),
        "baseline_argmax": {
            **{f"price_{k}": v for k, v in base_rates.items()},
            "macro_f1_overall": base_macro_f1,
        },
        "threshold_sweep": rows,
        "threshold_closing_gap_below_0_10": closing,
        "min_eo_gap_aspect_achievable": floor,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print()
    print(f"Aspect ha nguong: {args.aspect}   (n={base_rates['n']}, "
          f"n_Negative={base_rates['n_true_positive_class']})")
    print(f"Goc (argmax): TPR={base_rates['tpr']:.3f}  "
          f"FPR={base_rates['fpr']:.3f}  precision={base_rates['precision']:.3f}"
          f"  false positive={base_rates['n_false_positive']}"
          f"  macro-F1 toan cuc={base_macro_f1:.4f}")
    print()
    head = (f"{'nguong':>8}{'TPR':>8}{'FPR':>8}{'prec':>8}{'so FP':>8}"
            f"{'EOgap asp':>11}{'nhom yeu nhat':>16}{'macroF1':>10}")
    print(head)
    print("-" * len(head))
    for r in rows:
        # In thua moi 0,05 cho de doc; JSON van giu day du moi buoc quet.
        if round(r["threshold"] * 100) % 5 != 0:
            continue
        gap = "n/a" if r["eo_gap_aspect"] is None else f"{r['eo_gap_aspect']:.3f}"
        print(f"{r['threshold']:>8.3f}{r['price_tpr']:>8.3f}"
              f"{r['price_fpr']:>8.3f}{r['price_precision']:>8.3f}"
              f"{r['price_n_false_positive']:>8d}{gap:>11}"
              f"{str(r['worst_aspect']):>16}{r['macro_f1_overall']:>10.4f}")

    if floor:
        print()
        print(f"EO gap theo aspect thap nhat dat duoc: {floor['eo_gap_aspect']:.3f} "
              f"tai nguong {floor['threshold']:.3f}")
        print(f"  o day nhom yeu nhat khong con la {args.aspect} ma la "
              f"{floor['worst_aspect']} (TPR={floor['worst_aspect_tpr']:.3f}).")
        print("  Ha nguong tiep khong lam gap nho hon nua, chi lam hong them "
              f"{args.aspect}.")

    if closing:
        print()
        print("Nguong dau tien dua EO gap theo aspect xuong duoi 0,10: "
              f"{closing['threshold']:.3f}")
        print(f"  gia phai tra o {args.aspect}: "
              f"FPR {base_rates['fpr']:.3f} -> {closing['price_fpr']:.3f}, "
              f"precision {base_rates['precision']:.3f} -> "
              f"{closing['price_precision']:.3f}, "
              f"so false positive {base_rates['n_false_positive']} -> "
              f"{closing['price_n_false_positive']}")
        print(f"  macro-F1 toan cuc: {base_macro_f1:.4f} -> "
              f"{closing['macro_f1_overall']:.4f}")
    else:
        print()
        print("Khong nguong nao dua EO gap theo aspect xuong duoi 0,10.")
    print()
    print(f"Da luu -> {args.out}")


if __name__ == "__main__":
    main()
