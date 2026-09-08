"""
error_analysis.py — Phan loai co he thong nguyen nhan cac du doan sai.

Muc E3 cua docs/CHECKLIST.md xep o "Mot phan / Chua xac minh": da kiem tra bang
mat tren vai vi du, chua co quy trinh he thong de biet mo hinh bam vao dac trung
hop ly hay bam vao artifact. Hanh dong ghi kem muc do la "lay mau 50 du doan sai
va phan loai nguyen nhan". Script nay lam dung viec do, va lam bang code de con
so tai lap duoc thay vi phu thuoc vao viec ai doc.

Cau hoi trung tam khong phai "mo hinh sai bao nhieu" - macro-F1 da tra loi roi -
ma la "khi sai thi no dua vao cai gi". Voi mo hinh tuyen tinh, cau hoi do tra
loi duoc chinh xac: logit cua lop duoc du doan bang tong dong gop cua tung dac
trung cong intercept, nen co the do THANG ty le dong gop den tu moi loai dac
trung.

Bon loai dac trung, va y nghia dao duc cua tung loai:

  w:  n-gram tu     - bang chung noi dung thuc su. Day la thu ta muon.
  c:  n-gram ky tu  - van la noi dung, nhung o muc hinh thai; chap nhan duoc,
                      va chinh la ly do chon char_wb de chiu duoc teencode.
  a:  one-hot aspect- KHONG phai bang chung tu van ban. Neu dong gop cua no lon
                      thi mo hinh dang doan theo tien nghiem cua aspect chu
                      khong doc cau. Day la artifact dang lo nhat o day.
  artifact ky tu    - n-gram chi gom dau cau, khoang trang hoac chu so. Neu no
                      dan dau thi mo hinh dang bam vao vet dinh dang.

Cach dung:
    python src/error_analysis.py
    python src/error_analysis.py --n 100 --seed 7
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interpret import explain_text  # noqa: E402

# Tu noi tuong phan trong cau: dau hieu cua binh luan da sac thai, loai ma mot
# mo hinh khong dung ngu canh duoc du bao la se kho.
CONTRAST = re.compile(
    r"\b(nhưng|tuy nhiên|mặc dù|dù|cơ mà|có điều|ngoại trừ|chỉ tội|được cái)\b",
    re.I)


def classify_feature(name):
    """Phan loai mot ten dac trung tra ve tu explain_text."""
    kind, _, tok = name.partition(":")
    if kind == "a":
        return "aspect_onehot"
    if not re.search(r"[A-Za-zÀ-ỹ]", tok):
        return "artifact_ky_tu"
    return "ngram_tu" if kind == "w" else "ngram_ky_tu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/model_baseline.joblib")
    ap.add_argument("--pred", default="results/predictions_test.csv")
    ap.add_argument("--out", default="results/error_analysis.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    bundle = load(args.model)
    df = pd.read_csv(args.pred)
    wrong = df[df["polarity"] != df["pred"]]
    right = df[df["polarity"] == df["pred"]]
    n_wrong = len(wrong)

    sample = wrong.sample(n=min(args.n, n_wrong), random_state=args.seed)
    # NHOM DOI CHUNG. Khong co nhom nay thi cac con so ben duoi vo nghia: neu
    # 32/50 ca sai co one-hot aspect dan dau, cau hoi bat buoc phai hoi tiep la
    # ty le do o cac ca DUNG la bao nhieu. Chi phan chenh lech moi la phat hien.
    control = right.sample(n=min(args.n, len(right)), random_state=args.seed)

    rows = []
    for rec in sample.itertuples(index=False):
        exp = explain_text(bundle, str(rec.comment), rec.aspect, top_k=30)
        feats = exp["top_features"]

        # Tach dong gop theo loai. Dung tri tuyet doi vi ca dong gop am cung la
        # bang chung ma mo hinh dua vao.
        by_kind = Counter()
        for f in feats:
            by_kind[classify_feature(f["feature"])] += abs(f["contribution"])
        total = sum(by_kind.values()) or 1.0

        top = feats[0]
        rows.append({
            "aspect": rec.aspect,
            "true": rec.polarity,
            "pred": rec.pred,
            "confusion": f"{rec.polarity}->{rec.pred}",
            "n_words": int(rec.n_words),
            "co_tu_tuong_phan": bool(CONTRAST.search(str(rec.comment))),
            "prob_pred": exp["probabilities"][exp["prediction"]],
            "top_feature_kind": classify_feature(top["feature"]),
            "top_feature_contribution": top["contribution"],
            "ty_le_dong_gop": {k: round(v / total, 4) for k, v in by_kind.items()},
        })

    ctrl_rows = []
    for rec in control.itertuples(index=False):
        exp = explain_text(bundle, str(rec.comment), rec.aspect, top_k=30)
        feats = exp["top_features"]
        by_kind = Counter()
        for f in feats:
            by_kind[classify_feature(f["feature"])] += abs(f["contribution"])
        total = sum(by_kind.values()) or 1.0
        ctrl_rows.append({
            "co_tu_tuong_phan": bool(CONTRAST.search(str(rec.comment))),
            "prob_pred": exp["probabilities"][exp["prediction"]],
            "top_feature_kind": classify_feature(feats[0]["feature"]),
            "ty_le_dong_gop": {k: round(v / total, 4) for k, v in by_kind.items()},
        })

    def share(key, src=None):
        vals = [r["ty_le_dong_gop"].get(key, 0.0) for r in (src or rows)]
        return round(float(np.mean(vals)), 4)

    n = len(rows)
    nc = len(ctrl_rows)
    summary = {
        "n_sai_toan_bo": int(n_wrong),
        "n_du_doan_toan_bo": int(len(df)),
        "ty_le_sai": round(n_wrong / len(df), 4),
        "n_mau_phan_tich": n,
        "seed": args.seed,
        "phan_bo_nham_lan": dict(Counter(r["confusion"] for r in rows)
                                 .most_common()),
        "phan_bo_aspect": dict(Counter(r["aspect"] for r in rows).most_common()),
        "loai_dac_trung_dan_dau": dict(Counter(r["top_feature_kind"]
                                               for r in rows).most_common()),
        "ty_le_dong_gop_trung_binh": {
            "ngram_tu": share("ngram_tu"),
            "ngram_ky_tu": share("ngram_ky_tu"),
            "aspect_onehot": share("aspect_onehot"),
            "artifact_ky_tu": share("artifact_ky_tu"),
        },
        "co_tu_tuong_phan": sum(r["co_tu_tuong_phan"] for r in rows),
        "lien_quan_lop_neutral": sum(
            1 for r in rows if "Neutral" in r["confusion"]),
        "dao_cuc_nang": sum(
            1 for r in rows
            if {r["true"], r["pred"]} == {"Positive", "Negative"}),
        "prob_trung_binh": round(
            float(np.mean([r["prob_pred"] for r in rows])), 4),
    }
    summary["doi_chung_du_doan_dung"] = {
        "n_mau": nc,
        "loai_dac_trung_dan_dau": dict(Counter(r["top_feature_kind"]
                                               for r in ctrl_rows).most_common()),
        "ty_le_dong_gop_trung_binh": {
            k: share(k, ctrl_rows) for k in
            ("ngram_tu", "ngram_ky_tu", "aspect_onehot", "artifact_ky_tu")},
        "co_tu_tuong_phan": sum(r["co_tu_tuong_phan"] for r in ctrl_rows),
        "prob_trung_binh": round(
            float(np.mean([r["prob_pred"] for r in ctrl_rows])), 4),
    }

    # Kiem chung tren TOAN BO tap test, khong chi tren mau 50. Tu tuong phan
    # tuong quan voi do dai, va binh luan dai von da sai nhieu hon, nen phai
    # kiem soat do dai truoc khi noi tuong phan la nguyen nhan doc lap.
    full = df.copy()
    full["contrast"] = full["comment"].astype(str).str.contains(CONTRAST)
    full["err"] = full["polarity"] != full["pred"]
    full["q"] = pd.qcut(full["n_words"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    by_q = {}
    for q, sub in full.groupby("q", observed=True):
        a = sub[~sub["contrast"]]
        b = sub[sub["contrast"]]
        by_q[str(q)] = {
            "khoang_tu": [int(sub["n_words"].min()), int(sub["n_words"].max())],
            "ty_le_sai_khong_tuong_phan": round(float(a["err"].mean()), 4),
            "n_khong_tuong_phan": int(len(a)),
            "ty_le_sai_co_tuong_phan": round(float(b["err"].mean()), 4),
            "n_co_tuong_phan": int(len(b)),
        }

    out = {
        "purpose": ("Phan loai co he thong nguyen nhan du doan sai, phuc vu muc "
                    "E3 cua docs/CHECKLIST.md."),
        "tuong_phan_kiem_soat_do_dai": {
            "ghi_chu": ("Ty le sai theo tung tu phan vi do dai, tach theo viec "
                        "binh luan co tu tuong phan trong cau hay khong. Neu "
                        "chenh lech con giu trong TUNG tu phan vi thi tuong "
                        "phan la nguyen nhan doc lap chu khong phai bien thay "
                        "the cho do dai."),
            "toan_tap_test": {
                "ty_le_sai_khong_tuong_phan": round(
                    float(full[~full["contrast"]]["err"].mean()), 4),
                "ty_le_sai_co_tuong_phan": round(
                    float(full[full["contrast"]]["err"].mean()), 4),
            },
            "theo_tu_phan_vi": by_q,
        },
        "phuong_phap": (
            "Lay mau ngau nhien co seed tu cac du doan sai tren tap test. Voi "
            "moi mau, dung phan ra tuyen tinh chinh xac trong src/interpret.py "
            "de tach dong gop vao logit cua lop duoc du doan theo bon loai dac "
            "trung, roi tong hop."),
        "summary": summary,
        "cases": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    s = summary
    print()
    print(f"Sai {s['n_sai_toan_bo']}/{s['n_du_doan_toan_bo']} "
          f"({s['ty_le_sai']*100:.1f}%). Phan tich {n} mau, seed {args.seed}.")
    print()
    print("Nham lan pho bien nhat:")
    for k, v in list(s["phan_bo_nham_lan"].items())[:6]:
        print(f"  {k:<22}{v:>4}")
    c = s["doi_chung_du_doan_dung"]
    print()
    print(f"Lien quan lop Neutral : {s['lien_quan_lop_neutral']}/{n}")
    print(f"Dao cuc Pos<->Neg     : {s['dao_cuc_nang']}/{n}")
    print()
    print(f"{'':<34}{'SAI':>10}{'DUNG':>10}")
    print("-" * 54)
    print(f"{'Co tu tuong phan':<34}{s['co_tu_tuong_phan']:>7}/{n}"
          f"{c['co_tu_tuong_phan']:>7}/{nc}")
    print(f"{'Xac suat TB cua lop duoc chon':<34}"
          f"{s['prob_trung_binh']:>10.3f}{c['prob_trung_binh']:>10.3f}")
    print()
    print("Dac trung DAN DAU:")
    for k in ("aspect_onehot", "ngram_tu", "ngram_ky_tu", "artifact_ky_tu"):
        a = s["loai_dac_trung_dan_dau"].get(k, 0)
        b = c["loai_dac_trung_dan_dau"].get(k, 0)
        print(f"  {k:<32}{a:>7}/{n}{b:>7}/{nc}")
    print()
    print("Ty le dong gop trung binh vao logit lop duoc du doan:")
    for k, v in s["ty_le_dong_gop_trung_binh"].items():
        print(f"  {k:<32}{v*100:>9.1f}%"
              f"{c['ty_le_dong_gop_trung_binh'][k]*100:>9.1f}%")
    print()
    print("Tu tuong phan, kiem soat do dai (toan bo tap test):")
    print(f"  {'tu phan vi do dai':<24}{'khong tuong phan':>18}"
          f"{'co tuong phan':>16}")
    for q, v in out["tuong_phan_kiem_soat_do_dai"]["theo_tu_phan_vi"].items():
        lo, hi = v["khoang_tu"]
        print(f"  {q} ({lo}-{hi} tu)".ljust(26)
              + f"{v['ty_le_sai_khong_tuong_phan']*100:>10.1f}%"
                f" (n={v['n_khong_tuong_phan']})".ljust(8)
              + f"{v['ty_le_sai_co_tuong_phan']*100:>10.1f}%"
                f" (n={v['n_co_tuong_phan']})")
    print()
    print(f"Da luu -> {args.out}")


if __name__ == "__main__":
    main()
