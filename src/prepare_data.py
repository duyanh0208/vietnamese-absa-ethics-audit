"""
prepare_data.py — Chuyen UIT-ViSFD tu dinh dang raw sang dinh dang ABSA tuple.

Input : data/raw/{Train,Dev,Test}.csv
Output: data/processed/{train,dev,test}.jsonl

Moi dong output tuong ung MOT cap (comment, aspect) voi nhan polarity.
Day la dinh dang cho tac vu Aspect Category Sentiment Analysis (ACSA).

Nhan goc co dang: {BATTERY#Negative};{GENERAL#Positive};{OTHERS};
- Cap co polarity  -> mau huan luyen cho aspect do
- Nhan {OTHERS}    -> khong gan aspect nao, bo qua
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# 10 aspect theo dinh nghia goc cua UIT-ViSFD (Luc Phan et al., 2021)
ASPECTS = [
    "BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL",
    "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE",
]
POLARITIES = ["Positive", "Neutral", "Negative"]

LABEL_RE = re.compile(r"\{([^#{}]+)(?:#([^{}]+))?\}")


def parse_label(raw_label):
    """Tach chuoi nhan thanh dict {aspect: polarity}.

    Tra ve dict rong neu nhan trong hoac chi chua {OTHERS}.
    """
    if not isinstance(raw_label, str):
        return {}
    out = {}
    for aspect, polarity in LABEL_RE.findall(raw_label):
        aspect = aspect.strip()
        if aspect not in ASPECTS:
            continue  # bo qua OTHERS va cac nhan ngoai schema
        if polarity is None:
            continue
        polarity = polarity.strip()
        if polarity in POLARITIES:
            out[aspect] = polarity
    return out


def expand(df):
    """Mo rong moi comment thanh nhieu dong (comment, aspect, polarity)."""
    rows = []
    for rec in df.itertuples(index=False):
        pairs = parse_label(rec.label)
        if not pairs:
            continue
        comment = str(rec.comment).strip()
        if not comment:
            continue
        for aspect, polarity in pairs.items():
            rows.append({
                "id": int(rec.index),
                "comment": comment,
                "aspect": aspect,
                "polarity": polarity,
                "n_star": int(rec.n_star) if pd.notna(rec.n_star) else None,
                "n_chars": len(comment),
                "n_words": len(comment.split()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, fname in [("train", "Train.csv"), ("dev", "Dev.csv"), ("test", "Test.csv")]:
        src = raw_dir / fname
        if not src.exists():
            raise FileNotFoundError(
                f"Khong tim thay {src}. Chay scripts/download_data.sh truoc."
            )
        df = pd.read_csv(src)
        rows = expand(df)
        dst = out_dir / f"{split}.jsonl"
        with dst.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_comments = df.shape[0]
        print(f"{split:5s}: {n_comments:6d} comment -> {len(rows):6d} cap (comment, aspect)")


if __name__ == "__main__":
    main()
