"""
Kiem thu cho src/fairness_check.py.

Dung `unittest` cua thu vien chuan, khong them phu thuoc moi: requirements.txt
ghim chinh xac phien ban va la mot phan cua lap luan tai lap o chuong 7 cua
docs/ETHICS.md, nen them mot goi chi de chay test se lam yeu chinh lap luan do.

Chay:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairness_check import (  # noqa: E402
    MIN_GROUP_SIZE,
    _no_diacritic_ratio,
    _tpr,
    add_derived_groups,
    bootstrap_gap_ci,
    fairness_report,
)


def make_df(spec, seed=0):
    """Dung DataFrame co TPR biet truoc cho tung nhom.

    spec: {ten_nhom: (so_mau_Negative, so_mau_bat_duoc, so_mau_khac_lop)}
    """
    rows = []
    for g, (n_pos, n_hit, n_other) in spec.items():
        for i in range(n_pos):
            rows.append({"group": g, "polarity": "Negative",
                         "pred": "Negative" if i < n_hit else "Neutral"})
        for _ in range(n_other):
            rows.append({"group": g, "polarity": "Positive",
                         "pred": "Positive"})
    return pd.DataFrame(rows)


class TestTpr(unittest.TestCase):
    def test_tpr_co_ban(self):
        y_true = np.array(["Negative"] * 4 + ["Positive"] * 6)
        y_pred = np.array(["Negative"] * 3 + ["Neutral"] + ["Positive"] * 6)
        tpr, n = _tpr(y_true, y_pred, "Negative")
        self.assertAlmostEqual(tpr, 0.75)
        self.assertEqual(n, 4)

    def test_khong_co_mau_duong(self):
        y = np.array(["Positive", "Positive"])
        tpr, n = _tpr(y, y, "Negative")
        self.assertIsNone(tpr)
        self.assertEqual(n, 0)


class TestFairnessReport(unittest.TestCase):
    def test_gap_dung_bang_hieu_max_min(self):
        df = make_df({"A": (100, 90, 50),    # TPR 0,9
                      "B": (100, 60, 50),    # TPR 0,6
                      "C": (100, 75, 50)})   # TPR 0,75
        rep = fairness_report(df, "group", n_boot=0)
        self.assertAlmostEqual(rep["equal_opportunity_gap"], 0.30, places=9)
        self.assertEqual(rep["worst_group"]["group"], "B")
        self.assertEqual(rep["n_groups_evaluated"], 3)

    def test_nhom_duoi_nguong_bi_loai_khoi_gap(self):
        """Nhom nho co TPR cuc doan khong duoc phep keo gap len."""
        df = make_df({"A": (100, 90, 50),
                      "B": (100, 60, 50),
                      "TINY": (MIN_GROUP_SIZE - 1, 0, 10)})  # TPR 0 nhung nho
        rep = fairness_report(df, "group", n_boot=0)
        self.assertAlmostEqual(rep["equal_opportunity_gap"], 0.30, places=9)
        self.assertEqual(rep["n_groups_excluded_small"], 1)
        tiny = [r for r in rep["per_group"] if r["group"] == "TINY"][0]
        self.assertFalse(tiny["reliable"])

    def test_nguong_la_bien_chat_khong_phai_lon_hon(self):
        """Dung MIN_GROUP_SIZE mau thi HOP LE (>=), khong bi loai."""
        df = make_df({"A": (100, 90, 50),
                      "B": (MIN_GROUP_SIZE, MIN_GROUP_SIZE, 10)})
        rep = fairness_report(df, "group", n_boot=0)
        self.assertEqual(rep["n_groups_evaluated"], 2)
        self.assertEqual(rep["n_groups_excluded_small"], 0)

    def test_duoi_hai_nhom_thi_khong_ket_luan(self):
        df = make_df({"A": (100, 90, 50), "TINY": (5, 1, 5)})
        rep = fairness_report(df, "group", n_boot=0)
        self.assertIsNone(rep["equal_opportunity_gap"])
        self.assertIsNone(rep["worst_group"])
        self.assertIn("note", rep)

    def test_thieu_cot_thi_bao_loi_ro_rang(self):
        df = pd.DataFrame({"group": ["A"], "polarity": ["Negative"]})
        with self.assertRaises(KeyError):
            fairness_report(df, "group", n_boot=0)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.df = make_df({"A": (200, 180, 100),
                           "B": (200, 120, 100),
                           "C": (200, 150, 100)})

    def test_cung_seed_cho_cung_ket_qua(self):
        a = bootstrap_gap_ci(self.df, "group", ["A", "B", "C"], n_boot=200,
                             seed=42)
        b = bootstrap_gap_ci(self.df, "group", ["A", "B", "C"], n_boot=200,
                             seed=42)
        self.assertEqual(a["equal_opportunity_gap_ci"],
                         b["equal_opportunity_gap_ci"])

    def test_seed_khac_cho_ket_qua_khac(self):
        a = bootstrap_gap_ci(self.df, "group", ["A", "B", "C"], n_boot=200,
                             seed=1)
        b = bootstrap_gap_ci(self.df, "group", ["A", "B", "C"], n_boot=200,
                             seed=2)
        self.assertNotEqual(a["equal_opportunity_gap_ci"],
                            b["equal_opportunity_gap_ci"])

    def test_khoang_hop_le_va_bao_quanh_uoc_luong_diem(self):
        rep = fairness_report(self.df, "group", n_boot=500, boot_seed=42)
        lo, hi = rep["bootstrap"]["equal_opportunity_gap_ci"]
        self.assertLessEqual(lo, hi)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, rep["equal_opportunity_gap"])
        self.assertGreaterEqual(hi, rep["equal_opportunity_gap"])

    def test_mau_cang_lon_khoang_cang_hep(self):
        """Tinh chat thong ke co ban; neu sai la cach resample da hong."""
        small = make_df({"A": (40, 36, 20), "B": (40, 24, 20)})
        large = make_df({"A": (4000, 3600, 2000), "B": (4000, 2400, 2000)})
        w = []
        for df in (small, large):
            ci = bootstrap_gap_ci(df, "group", ["A", "B"], n_boot=400,
                                  seed=42)["equal_opportunity_gap_ci"]
            w.append(ci[1] - ci[0])
        self.assertGreater(w[0], w[1] * 3)

    def test_nhom_nho_van_co_CI_va_CI_do_rat_rong(self):
        """CI cua nhom qua nho phai rong den muc vo dung - do la bang chung
        cho su ton tai cua MIN_GROUP_SIZE."""
        df = make_df({"A": (200, 180, 100), "B": (200, 120, 100),
                      "TINY": (6, 4, 5)})
        rep = fairness_report(df, "group", n_boot=500, boot_seed=42)
        ci = rep["bootstrap"]["per_group_tpr_ci"]["TINY"]
        self.assertGreater(ci[1] - ci[0], 0.4)

    def test_duoi_hai_nhom_thi_tra_ve_None_kem_ghi_chu(self):
        out = bootstrap_gap_ci(self.df, "group", ["A"], n_boot=100, seed=42)
        self.assertIsNone(out["equal_opportunity_gap_ci"])
        self.assertIn("note", out)

    def test_tat_bootstrap(self):
        rep = fairness_report(self.df, "group", n_boot=0)
        self.assertIsNone(rep["bootstrap"])


class TestNoDiacriticRatio(unittest.TestCase):
    def test_khong_dau_hoan_toan(self):
        self.assertAlmostEqual(_no_diacritic_ratio("may nay rat tot"), 1.0)

    def test_co_dau_hoan_toan(self):
        self.assertAlmostEqual(_no_diacritic_ratio("máy đẹp mượt"), 0.0)

    def test_tron_lan(self):
        # "may" khong dau, "đẹp" co dau -> 1/2
        self.assertAlmostEqual(_no_diacritic_ratio("may đẹp"), 0.5)

    def test_khong_co_tu_nao(self):
        self.assertIsNone(_no_diacritic_ratio("123 !!! ..."))
        self.assertIsNone(_no_diacritic_ratio(""))

    def test_bo_qua_chu_so_va_dau_cau(self):
        self.assertAlmostEqual(_no_diacritic_ratio("tot 100% !!!"), 1.0)


class TestDerivedGroups(unittest.TestCase):
    def test_sinh_du_ba_cot_dan_xuat(self):
        df = pd.DataFrame({
            "n_star": [1, 2, 3, 4, 5] * 4,
            "n_words": list(range(1, 21)),
            "comment": ["máy đẹp", "may dep", "pin trâu", "pin trau"] * 5,
        })
        out = add_derived_groups(df)
        for col in ("star_bucket", "length_bucket", "writing_bucket"):
            self.assertIn(col, out.columns)
            self.assertFalse(out[col].isna().any())

    def test_khong_co_cot_nguon_thi_khong_sinh_cot_dan_xuat(self):
        out = add_derived_groups(pd.DataFrame({"x": [1, 2, 3]}))
        for col in ("star_bucket", "length_bucket", "writing_bucket"):
            self.assertNotIn(col, out.columns)

    def test_khong_sua_dataframe_goc(self):
        df = pd.DataFrame({"n_star": [1, 5], "n_words": [3, 9],
                           "comment": ["a", "b"]})
        cols_truoc = list(df.columns)
        add_derived_groups(df)
        self.assertEqual(list(df.columns), cols_truoc)


if __name__ == "__main__":
    unittest.main()
