"""
Kiem thu cho src/sensitivity_threshold.py.

Trong tam la hai bat bien co the am tham hong ma khong lam script bao loi:

  1. `apply_threshold` chi duoc dong vao cac dong thuoc target_mask. Neu no ro ri
     ra ngoai thi moi con so "EO gap toan cuc" o muc 6.2 cua docs/ETHICS.md deu
     sai ma khong co dau hieu gi.
  2. So du doan Negative trong nhom muc tieu phai khong tang khi nguong tang.
     Neu quan he nay dut thi bang quet nguong khong con doc duoc.

Chay:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensitivity_threshold import (  # noqa: E402
    apply_threshold,
    group_rates,
)

CLASSES = ["Negative", "Neutral", "Positive"]


def proba(neg, neu, pos):
    return np.array([[neg, neu, pos]])


class TestApplyThreshold(unittest.TestCase):
    def setUp(self):
        # 4 dong: 2 trong nhom muc tieu, 2 ngoai.
        self.proba = np.array([
            [0.30, 0.50, 0.20],   # muc tieu
            [0.10, 0.20, 0.70],   # muc tieu
            [0.30, 0.50, 0.20],   # ngoai
            [0.90, 0.05, 0.05],   # ngoai
        ])
        self.base = np.array(["Neutral", "Positive", "Neutral", "Negative"])
        self.mask = np.array([True, True, False, False])

    def test_nguong_0_thi_moi_dong_muc_tieu_thanh_Negative(self):
        out = apply_threshold(self.proba, CLASSES, self.base, self.mask, 0.0)
        self.assertEqual(list(out[:2]), ["Negative", "Negative"])

    def test_nguong_tren_1_thi_khong_dong_nao_thanh_Negative(self):
        out = apply_threshold(self.proba, CLASSES, self.base, self.mask, 1.01)
        self.assertNotIn("Negative", list(out[:2]))

    def test_khi_khong_vuot_nguong_thi_chon_lop_tot_nhat_trong_hai_lop_con_lai(self):
        out = apply_threshold(self.proba, CLASSES, self.base, self.mask, 0.99)
        # dong 0: Neutral 0,50 > Positive 0,20 -> Neutral
        # dong 1: Positive 0,70 > Neutral 0,20 -> Positive
        self.assertEqual(out[0], "Neutral")
        self.assertEqual(out[1], "Positive")

    def test_khong_bao_gio_de_trong_du_doan(self):
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = apply_threshold(self.proba, CLASSES, self.base, self.mask, t)
            self.assertTrue(all(v in CLASSES for v in out))

    def test_dong_ngoai_nhom_muc_tieu_khong_bao_gio_bi_doi(self):
        """Bat bien quan trong nhat: ro ri ra ngoai mask se lam sai moi con so
        EO gap toan cuc trong bang o muc 6.2 ma khong co dau hieu gi."""
        for t in np.linspace(0.0, 1.0, 21):
            out = apply_threshold(self.proba, CLASSES, self.base, self.mask, t)
            self.assertEqual(list(out[2:]), list(self.base[2:]))

    def test_khong_sua_mang_base_pred_goc(self):
        base_copy = self.base.copy()
        apply_threshold(self.proba, CLASSES, self.base, self.mask, 0.1)
        self.assertEqual(list(self.base), list(base_copy))

    def test_so_du_doan_Negative_khong_tang_khi_nguong_tang(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet([1, 1, 1], size=300)
        base = np.array(CLASSES)[p.argmax(axis=1)]
        mask = np.ones(300, dtype=bool)
        truoc = None
        for t in np.linspace(0.0, 1.0, 41):
            out = apply_threshold(p, CLASSES, base, mask, t)
            n = int((out == "Negative").sum())
            if truoc is not None:
                self.assertLessEqual(n, truoc)
            truoc = n

    def test_thu_tu_lop_khong_anh_huong_ket_qua(self):
        """clf.classes_ do sklearn sap xep; ham khong duoc gia dinh Negative
        nam o vi tri 0."""
        classes_b = ["Neutral", "Negative", "Positive"]
        proba_b = self.proba[:, [1, 0, 2]]
        a = apply_threshold(self.proba, CLASSES, self.base, self.mask, 0.25)
        b = apply_threshold(proba_b, classes_b, self.base, self.mask, 0.25)
        self.assertEqual(list(a), list(b))


class TestGroupRates(unittest.TestCase):
    def test_tinh_dung_tpr_fpr_precision(self):
        y_true = np.array(["Negative"] * 4 + ["Positive"] * 6)
        #    bat duoc 3/4 Negative; bao nham 2/6 khong-Negative
        y_pred = np.array(["Negative"] * 3 + ["Neutral"]
                          + ["Negative"] * 2 + ["Positive"] * 4)
        r = group_rates(y_true, y_pred)
        self.assertEqual(r["n"], 10)
        self.assertEqual(r["n_true_positive_class"], 4)
        self.assertAlmostEqual(r["tpr"], 3 / 4)
        self.assertAlmostEqual(r["fpr"], 2 / 6)
        self.assertAlmostEqual(r["precision"], 3 / 5)
        self.assertEqual(r["n_false_positive"], 2)
        self.assertEqual(r["n_true_pos_caught"], 3)

    def test_khong_co_mau_duong_thi_tpr_None(self):
        y = np.array(["Positive", "Neutral"])
        r = group_rates(y, y)
        self.assertIsNone(r["tpr"])

    def test_khong_du_doan_Negative_nao_thi_precision_None(self):
        y_true = np.array(["Negative", "Positive"])
        y_pred = np.array(["Neutral", "Positive"])
        r = group_rates(y_true, y_pred)
        self.assertIsNone(r["precision"])
        self.assertAlmostEqual(r["tpr"], 0.0)


class TestTaiLapVoiMoHinhThat(unittest.TestCase):
    """Kiem thu tich hop: bo qua neu chua huan luyen mo hinh."""

    def setUp(self):
        self.model = ROOT / "results" / "model_baseline.joblib"
        self.pred = ROOT / "results" / "predictions_test.csv"
        self.data = ROOT / "data" / "processed" / "test.jsonl"
        if not (self.model.exists() and self.pred.exists()
                and self.data.exists()):
            self.skipTest("chua co model/du lieu; chay pipeline truoc")

    def test_argmax_tai_lap_dung_predictions_test_csv(self):
        """build_test_matrix chi transform, khong khop lai vectorizer. Neu no
        vo tinh khop lai thi du doan se lech khoi predictions_test.csv."""
        from joblib import load
        from sensitivity_threshold import build_test_matrix

        bundle = load(self.model)
        test = pd.read_json(self.data, lines=True)
        X = build_test_matrix(bundle, test)
        pred = bundle["clf"].predict(X)
        goc = pd.read_csv(self.pred)["pred"].to_numpy()
        self.assertEqual(len(pred), len(goc))
        self.assertTrue((pred == goc).all())


if __name__ == "__main__":
    unittest.main()
