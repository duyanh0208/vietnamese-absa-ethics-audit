"""
fairness_check.py — Kiem toan cong bang cho mo hinh ABSA.

Metric chinh: EQUAL OPPORTUNITY GAP tren lop Negative.

    TPR_g = P(pred = Negative | true = Negative, group = g)
    EO_gap = max_g TPR_g  -  min_g TPR_g

Ly do chon metric nay cho bai toan ABSA/social listening duoc trinh bay
day du trong docs/ETHICS.md chuong 3. Tom tat: tac hai cu the trong he thong
nay la BO SOT phan anh tieu cuc cua mot nhom nguoi dung nao do. Do chinh
la false negative tren lop Negative, tuc la sai lech ve TPR - dung pham vi
do cua equal opportunity (Hardt et al., 2016). Demographic parity KHONG
phu hop vi ty le phan nan thuc su khac nhau giua cac aspect; ep bang nhau
se lam sai lech tin hieu.

Metric phu:
  - Macro-F1 gap giua cac nhom (do lech chat luong tong the)
  - Selection rate (ty le du doan Negative) - de doi chieu, khong phai muc tieu

DO BAT DINH. EO gap duoc bao cao kem khoang tin cay bootstrap 95%. Ly do:
TPR cua mot nhom nho duoc uoc luong tren rat it mau (aspect PRICE chi co 79
mau Negative tren tap test), nen mot con so gap tran trui goi y do chinh xac
ma phep do khong co. Voi n=79, mot mau doi nhan da lam TPR dich 1/79 = 0,013.
Xem `bootstrap_gap_ci` de biet cach lay mau va cac gioi han cua no.

Cach dung:
    python src/fairness_check.py --pred results/predictions_test.csv

Hoac import nhu mot ham:
    from fairness_check import fairness_report
    rep = fairness_report(df, group_col="aspect", positive_class="Negative")
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

MIN_GROUP_SIZE = 30  # nhom nho hon nguong nay bi loai khoi tinh gap

_DIACRITIC_RE = re.compile(
    "[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵđ]", re.I)
_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")

N_BOOTSTRAP = 2000    # so lan resample mac dinh
BOOTSTRAP_SEED = 42   # co dinh de ket qua tai lap duoc
CI_LEVEL = 0.95       # khoang percentile hai phia


def _tpr(y_true, y_pred, positive_class):
    """True positive rate (recall) cho mot lop cu the."""
    mask = y_true == positive_class
    n = int(mask.sum())
    if n == 0:
        return None, 0
    return float((y_pred[mask] == positive_class).mean()), n


def fairness_report(df, group_col, y_true_col="polarity", y_pred_col="pred",
                    positive_class="Negative", min_size=MIN_GROUP_SIZE,
                    n_boot=N_BOOTSTRAP, boot_seed=BOOTSTRAP_SEED):
    """Tinh chi so cong bang theo tung nhom cua `group_col`.

    Tham so
    -------
    df : DataFrame co it nhat 3 cot: group_col, y_true_col, y_pred_col
    group_col : ten cot dinh nghia nhom (aspect, star_bucket, length_bucket...)
    positive_class : lop duoc coi la "can phat hien" khi tinh TPR
    min_size : nhom co it hon so mau nay bi danh dau unreliable va khong
               tham gia tinh gap (tranh gap gia do nhieu thong ke)
    n_boot : so lan resample de uoc luong khoang tin cay; 0 de tat
    boot_seed : seed cua bootstrap, co dinh de tai lap

    Tra ve
    ------
    dict gom: per_group, equal_opportunity_gap, macro_f1_gap, worst_group,
    bootstrap (khoang tin cay 95% cho gap va cho TPR tung nhom)
    """
    for col in (group_col, y_true_col, y_pred_col):
        if col not in df.columns:
            raise KeyError(f"Thieu cot '{col}' trong du lieu du doan.")

    rows = []
    for g, sub in df.groupby(group_col, dropna=False):
        y_true = sub[y_true_col].to_numpy()
        y_pred = sub[y_pred_col].to_numpy()
        tpr, n_pos = _tpr(y_true, y_pred, positive_class)
        rows.append({
            "group": str(g),
            "n": int(len(sub)),
            f"n_{positive_class.lower()}": n_pos,
            "tpr": tpr,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0)),
            "selection_rate": float((y_pred == positive_class).mean()),
            "reliable": bool(n_pos >= min_size),
        })

    per_group = sorted(rows, key=lambda r: (r["tpr"] is None, r["tpr"]))
    usable = [r for r in per_group if r["reliable"] and r["tpr"] is not None]

    if len(usable) < 2:
        return {
            "group_col": group_col,
            "positive_class": positive_class,
            "per_group": per_group,
            "equal_opportunity_gap": None,
            "macro_f1_gap": None,
            "worst_group": None,
            "bootstrap": bootstrap_gap_ci(
                df, group_col, [], y_true_col, y_pred_col, positive_class,
                n_boot=n_boot, seed=boot_seed,
            ) if n_boot else None,
            "note": (f"Duoi {min_size} mau lop {positive_class} o hau het cac "
                     f"nhom; khong du du lieu de ket luan."),
        }

    tprs = [r["tpr"] for r in usable]
    f1s = [r["macro_f1"] for r in usable]
    worst = min(usable, key=lambda r: r["tpr"])

    boot = bootstrap_gap_ci(
        df, group_col, [r["group"] for r in usable], y_true_col, y_pred_col,
        positive_class, n_boot=n_boot, seed=boot_seed,
    ) if n_boot else None

    if boot:
        ci = boot.get("per_group_tpr_ci", {})
        for r in per_group:
            r["tpr_ci"] = ci.get(r["group"])

    return {
        "group_col": group_col,
        "positive_class": positive_class,
        "n_groups_evaluated": len(usable),
        "n_groups_excluded_small": len(per_group) - len(usable),
        "per_group": per_group,
        "equal_opportunity_gap": float(max(tprs) - min(tprs)),
        "macro_f1_gap": float(max(f1s) - min(f1s)),
        "worst_group": {"group": worst["group"], "tpr": worst["tpr"],
                        "n": worst["n"]},
        "bootstrap": boot,
    }


def bootstrap_gap_ci(df, group_col, eligible_groups, y_true_col="polarity",
                     y_pred_col="pred", positive_class="Negative",
                     n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
                     ci_level=CI_LEVEL):
    """Khoang tin cay bootstrap cho EO gap va cho TPR cua tung nhom.

    Cach lay mau
    ------------
    Resample CO HOAN LAI o MUC MAU: moi lan lap rut lai len(df) dong tu tap
    danh gia, khong phai rut lai danh sach nhom. Day la don vi dung, vi cai
    ngau nhien can mo phong la "neu tap test la mot mau khac cua cung phan bo"
    chu khong phai "neu ta chon nhom khac de do".

    Rut lai toan bo dong (khong chi cac dong mang nhan positive_class) khien
    kich thuoc lop Negative trong tung nhom cung dao dong dung nhu ngoai doi.
    Hien thuc bang multinomial: `w` la so lan moi dong duoc rut, tuong duong
    voi rut len(df) chi so co hoan lai nhung tinh nhanh hon nhieu.

    Tap nhom duoc CO DINH theo `eligible_groups` lay tu uoc luong diem, khong
    tinh lai nguong MIN_GROUP_SIZE trong tung lan lap. Neu de nguong tu do bien
    thien thi gap se duoc tinh tren nhung tap nhom khac nhau giua cac lan lap
    va khoang thu duoc tron lan hai nguon bien thien khac han nhau.

    Gioi han can biet khi doc ket qua
    ---------------------------------
    gap = max_g TPR_g - min_g TPR_g la mot thong ke LECH LEN: max cua cac uoc
    luong nhieu lon hon max cua gia tri that. Khoang percentile o day khong
    hieu chinh do lech do, nen no tra loi "gap do duoc dao dong bao nhieu",
    khong tra loi "gap that nam trong khoang nay voi xac suat 95%". Voi muc
    dich cua bao cao - cho thay con so gap khong nen doc den chu so thu ba -
    muc nay la du.

    Tra ve
    ------
    dict, hoac dict co `equal_opportunity_gap_ci = None` kem `note` neu khong
    du du lieu.
    """
    lo_q = (1.0 - ci_level) / 2.0 * 100.0
    hi_q = (1.0 + ci_level) / 2.0 * 100.0
    base = {
        "n_bootstrap": int(n_boot),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "resample_unit": "sample",
    }

    if len(eligible_groups) < 2:
        return {
            **base,
            "equal_opportunity_gap_ci": None,
            "per_group_tpr_ci": {},
            "note": ("Duoi hai nhom du kich thuoc; khong tinh duoc khoang tin "
                     "cay cho gap."),
        }

    y_true = df[y_true_col].to_numpy()
    y_pred = df[y_pred_col].to_numpy()
    groups = df[group_col].astype(str).to_numpy()

    is_pos = y_true == positive_class
    is_hit = is_pos & (y_pred == positive_class)

    # Tinh CI cho MOI nhom co it nhat 1 mau positive_class, ke ca nhom bi loai
    # khoi phep tinh gap: mot khoang rong den vo nghia chinh la bang chung cho
    # thay vi sao nhom do bi loai.
    all_groups = [g for g in pd.unique(groups) if is_pos[groups == g].sum() > 0]
    codes = {g: i for i, g in enumerate(all_groups)}
    n_groups = len(all_groups)
    if n_groups == 0:
        return {
            **base,
            "equal_opportunity_gap_ci": None,
            "per_group_tpr_ci": {},
            "note": f"Khong co mau nao thuoc lop {positive_class}.",
        }

    g_idx = np.array([codes.get(g, -1) for g in groups])
    keep = g_idx >= 0
    g_idx_pos = g_idx[keep & is_pos]
    hit_pos = is_hit[keep & is_pos].astype(float)

    elig_idx = np.array([codes[g] for g in eligible_groups if g in codes])
    if len(elig_idx) < 2:
        return {
            **base,
            "equal_opportunity_gap_ci": None,
            "per_group_tpr_ci": {},
            "note": ("Cac nhom du kich thuoc khong con mau positive_class nao; "
                     "khong tinh duoc khoang tin cay."),
        }

    n = len(df)
    rng = np.random.default_rng(seed)
    tpr_draws = np.full((n_boot, n_groups), np.nan)

    # Trong so multinomial cho TOAN BO dong, sau do chi lay phan positive_class:
    # so mau Negative cua tung nhom vi vay cung dao dong giua cac lan lap.
    idx_of_row = np.flatnonzero(keep & is_pos)
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        w_pos = w[idx_of_row].astype(float)
        denom = np.bincount(g_idx_pos, weights=w_pos, minlength=n_groups)
        numer = np.bincount(g_idx_pos, weights=w_pos * hit_pos,
                            minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            tpr_draws[b] = np.where(denom > 0, numer / denom, np.nan)

    per_group_ci = {}
    for g, i in codes.items():
        col = tpr_draws[:, i]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            per_group_ci[g] = None
            continue
        per_group_ci[g] = [float(np.percentile(col, lo_q)),
                           float(np.percentile(col, hi_q))]

    sub = tpr_draws[:, elig_idx]
    n_valid = (~np.isnan(sub)).sum(axis=1)
    ok = n_valid >= 2
    if not ok.any():
        return {
            **base,
            "equal_opportunity_gap_ci": None,
            "per_group_tpr_ci": per_group_ci,
            "note": ("Khong lan resample nao giu lai duoc hai nhom co mau "
                     f"{positive_class}; khong tinh duoc khoang tin cay."),
        }

    gaps = np.nanmax(sub[ok], axis=1) - np.nanmin(sub[ok], axis=1)
    out = {
        **base,
        "n_groups_in_gap": int(len(elig_idx)),
        "n_replicates_used": int(ok.sum()),
        "n_replicates_dropped": int((~ok).sum()),
        "equal_opportunity_gap_ci": [float(np.percentile(gaps, lo_q)),
                                     float(np.percentile(gaps, hi_q))],
        "equal_opportunity_gap_bootstrap_mean": float(gaps.mean()),
        "per_group_tpr_ci": per_group_ci,
    }
    if out["n_replicates_dropped"]:
        out["note"] = (f"{out['n_replicates_dropped']}/{n_boot} lan resample bi "
                       "bo vi con duoi hai nhom co mau du de tinh TPR.")
    return out


def _no_diacritic_ratio(text):
    """Ty le tu KHONG mang dau thanh/dau phu trong mot binh luan.

    Day la proxy cho mot dang viet khong chuan cu the: viet tieng Viet khong
    dau. Chon dai luong nay vi no do duoc truc tiep tu van ban, khong can tu
    dien, va vi viet khong dau la dang lech chuan pho bien nhat trong du lieu
    binh luan thuong mai dien tu.

    Gioi han phai noi ro: proxy KHONG bat duoc teencode co dau ("dth", "ntn"),
    viet tat, hay loi chinh ta co dau. No cung dem ca nhung tu von khong co
    dau ("ok", "pin", "ram", "iphone") la khong dau, nen gia tri tuyet doi cua
    ty le khong co y nghia - chi co thu tu giua cac binh luan la dung.
    """
    toks = _TOKEN_RE.findall(str(text))
    if not toks:
        return None
    return sum(0 if _DIACRITIC_RE.search(w) else 1 for w in toks) / len(toks)


def add_derived_groups(df):
    """Tao cac cot nhom dan xuat tu metadata co san trong UIT-ViSFD."""
    out = df.copy()
    if "n_star" in out.columns:
        out["star_bucket"] = pd.cut(
            out["n_star"], bins=[0, 2, 3, 5],
            labels=["1-2 sao", "3 sao", "4-5 sao"],
        ).astype(str)
    if "n_words" in out.columns:
        # Chia theo tu phan vi cua chinh tap danh gia thay vi nguong co dinh.
        # Nguong co dinh (vi du 10 va 25 tu) khong phu hop voi bo du lieu nay:
        # phan vi 25 da la 23 tu, nen bucket "ngan" chi con vai chuc mau va bi
        # loai khoi phep tinh gap, khien phep kiem toan chi con so sanh 2 nhom.
        q = pd.qcut(out["n_words"], 4, labels=False, duplicates="drop")
        edges = out.groupby(q, observed=True)["n_words"].agg(["min", "max"])
        names = {i: f"Q{i+1} ({r['min']}-{r['max']} tu)"
                 for i, r in edges.iterrows()}
        out["length_bucket"] = q.map(names).astype(str)
    if "comment" in out.columns:
        # Chieu nay kiem chung mot khang dinh cong bang o README: dung char
        # n-gram thay vi tach tu de "nguoi dung viet khong chuan khong bi mo
        # hinh hieu sai nhieu hon". Khang dinh do can duoc DO chu khong chi
        # duoc neu, nen o day no tro thanh mot chieu phan nhom that.
        nd = out["comment"].map(_no_diacritic_ratio)
        q = pd.qcut(nd, 4, labels=False, duplicates="drop")
        edges = out.assign(_nd=nd).groupby(q, observed=True)["_nd"].agg(
            ["min", "max"])
        names = {i: f"Q{i+1} ({r['min']*100:.0f}-{r['max']*100:.0f}% tu khong dau)"
                 for i, r in edges.iterrows()}
        out["writing_bucket"] = q.map(names).astype(str)
    return out


def print_report(rep):
    print(f"\n=== Nhom theo: {rep['group_col']}  "
          f"(lop muc tieu: {rep['positive_class']}) ===")
    header = (f"{'nhom':<18}{'n':>7}{'n_neg':>7}{'TPR':>9}"
              f"{'TPR CI 95%':>18}{'macroF1':>10}{'sel_rate':>10}")
    print(header)
    print("-" * len(header))
    for r in rep["per_group"]:
        tpr = "  n/a" if r["tpr"] is None else f"{r['tpr']:.3f}"
        flag = "" if r["reliable"] else "  (nho)"
        key = [k for k in r if k.startswith("n_")][0]
        ci = r.get("tpr_ci")
        ci_s = "-" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"{r['group']:<18}{r['n']:>7}{r[key]:>7}{tpr:>9}{ci_s:>18}"
              f"{r['macro_f1']:>10.3f}{r['selection_rate']:>10.3f}{flag}")
    if rep["equal_opportunity_gap"] is None:
        print(rep.get("note", ""))
        return

    boot = rep.get("bootstrap") or {}
    gap_ci = boot.get("equal_opportunity_gap_ci")
    gap_ci_s = "" if not gap_ci else f"   CI 95% [{gap_ci[0]:.3f}, {gap_ci[1]:.3f}]"
    print(f"\nEqual opportunity gap : {rep['equal_opportunity_gap']:.3f}{gap_ci_s}")
    print(f"Macro-F1 gap          : {rep['macro_f1_gap']:.3f}")
    print(f"Nhom yeu nhat         : {rep['worst_group']['group']} "
          f"(TPR={rep['worst_group']['tpr']:.3f})")
    if gap_ci:
        print(f"Bootstrap             : {boot['n_bootstrap']} lan resample o muc mau, "
              f"seed {boot['seed']}, {boot['n_replicates_used']} lan hop le")
    if boot.get("note"):
        print(f"Ghi chu bootstrap     : {boot['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/predictions_test.csv")
    ap.add_argument("--out", default="results/fairness_report.json")
    ap.add_argument("--positive-class", default="Negative")
    ap.add_argument("--groups", nargs="*",
                    default=["aspect", "star_bucket", "length_bucket",
                             "writing_bucket"])
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                    help="so lan resample; 0 de tat khoang tin cay")
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()

    df = add_derived_groups(pd.read_csv(args.pred))

    reports = {}
    for col in args.groups:
        if col not in df.columns:
            print(f"[bo qua] khong co cot '{col}'")
            continue
        rep = fairness_report(df, col, positive_class=args.positive_class,
                              n_boot=args.n_bootstrap,
                              boot_seed=args.bootstrap_seed)
        print_report(rep)
        reports[col] = rep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDa luu bao cao -> {args.out}")


if __name__ == "__main__":
    main()
