"""
scan_pii.py — Quet thong tin dinh danh ca nhan trong noi dung binh luan.

Muc 2.3 cua docs/ETHICS.md ghi rang truoc khi trich bat ky vi du nao vao luan
van thi buoc quet PII la BAT BUOC, va muc A4 cua docs/CHECKLIST.md xep muc nay
la "Chua xac minh" - chua quet nen chua biet trong du lieu co gi. Script nay
thuc hien buoc do.

Pham vi. Script quet cac dinh danh co the bat bang bieu thuc chinh quy: so dien
thoai, email, URL, dinh danh mang xa hoi, va day so dai co the la ma don hang.
Script KHONG bat duoc ten rieng, dia chi, hay bien the viet lach cua so dien
thoai (vi du "khong chin tam..."), va gioi han do duoc ghi thang vao ket qua
thay vi de nguoi doc tu suy ra.

NGUYEN TAC AN TOAN DAU RA. results/pii_scan.json nam trong repository, nen
script KHONG BAO GIO ghi noi dung binh luan tho vao do. Moi chuoi khop deu bi
che (chu so -> #, chu cai -> x) truoc khi ghi. Neu can xem ban goc de tham dinh
thu cong, dung --show-raw; co dinh la chi in ra man hinh, khong ghi ra tep.

Cach dung:
    python src/scan_pii.py
    python src/scan_pii.py --show-raw --limit 5
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Moi mau kem mot ghi chu ve do tin cay, vi ty le duong tinh gia rat khac nhau
# giua cac mau va gop chung lai thanh mot con so se gay hieu nham.
PATTERNS = {
    "email": (
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "Do tin cay cao. Rat it kha nang duong tinh gia.",
    ),
    "url": (
        re.compile(r"\b(?:https?://|www\.)\S+", re.I),
        "Do tin cay cao. Ban than URL chua chac la PII nhung co the tro toi "
        "trang ca nhan.",
    ),
    "social_handle": (
        re.compile(r"\b(?:fb\.com/|facebook\.com/|zalo\s*[:.]?\s*\d|@[A-Za-z]"
                   r"[\w.]{3,})", re.I),
        "Do tin cay trung binh. '@' con duoc dung de tag thong thuong.",
    ),
    "phone_vn": (
        re.compile(r"(?<!\d)(?:\+?84|0)\d{8,10}(?!\d)"),
        "Do tin cay TRUNG BINH THAP. Day so lien tuc 9-11 chu so bat dau bang "
        "0 cung khop voi gia tien viet lien (vi du '0' dung dau mot day gia). "
        "Bat buoc tham dinh thu cong.",
    ),
    "long_digits": (
        re.compile(r"(?<!\d)\d{8,}(?!\d)"),
        "Do tin cay THAP. Chu yeu la gia tien (vi du 20000000). Giu lai vi ma "
        "don hang cung roi vao mau nay; can loc tay.",
    ),
}


def mask(s):
    """Che chuoi khop truoc khi ghi ra tep nam trong repository."""
    return re.sub(r"[A-Za-zÀ-ỹ]", "x", re.sub(r"\d", "#", s))


def scan_split(df, patterns):
    hits = defaultdict(list)
    n_flagged_rows = set()
    for rec in df.itertuples(index=False):
        text = str(rec.comment)
        for name, (rx, _) in patterns.items():
            for m in rx.findall(text):
                m = m if isinstance(m, str) else m[0]
                hits[name].append(m)
                n_flagged_rows.add(rec.id)
    return hits, n_flagged_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results/pii_scan.json")
    ap.add_argument("--limit", type=int, default=8,
                    help="so vi du (da che) luu cho moi mau")
    ap.add_argument("--show-raw", action="store_true",
                    help="in ban goc ra man hinh de tham dinh tay; "
                         "KHONG ghi vao tep")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path(args.data_dir)
    per_split = {}
    grand = Counter()
    raw_examples = defaultdict(list)

    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Khong tim thay {path}. Chay prepare_data.py truoc.")
        df = pd.read_json(path, lines=True)
        # Mot binh luan sinh ra nhieu dong (mot dong moi aspect); de dem theo
        # BINH LUAN chu khong theo cap, bo trung theo id.
        uniq = df.drop_duplicates(subset="id")
        hits, flagged = scan_split(uniq, PATTERNS)

        counts = {k: len(v) for k, v in hits.items()}
        grand.update(counts)
        for k, v in hits.items():
            raw_examples[k].extend(v)

        per_split[split] = {
            "n_comments": int(len(uniq)),
            "n_comments_flagged": len(flagged),
            "pct_comments_flagged": round(100.0 * len(flagged) / len(uniq), 3),
            "n_matches_by_pattern": counts,
        }
        print(f"{split:5s}: {len(uniq):5d} binh luan, "
              f"{len(flagged):4d} bi gan co ({per_split[split]['pct_comments_flagged']}%)")

    out = {
        "purpose": ("Quet PII trong noi dung binh luan, phuc vu muc 2.3 cua "
                    "docs/ETHICS.md va muc A4 cua docs/CHECKLIST.md."),
        "note_output_safety": (
            "Tep nay nam trong repository nen KHONG chua noi dung binh luan tho. "
            "Moi chuoi khop deu da bi che: chu so -> #, chu cai -> x."),
        "patterns": {k: {"regex": rx.pattern, "do_tin_cay": note}
                     for k, (rx, note) in PATTERNS.items()},
        "limitations": [
            "Khong bat duoc ten rieng va dia chi: can NER tieng Viet, va chinh "
            "viec chay NER tren du lieu nay lai la mot buoc xu ly them can can "
            "nhac rieng.",
            "Khong bat duoc so dien thoai viet bang chu hoac chen ky tu la "
            "(vi du '09xx xxx xxx', 'khong chin ...').",
            "Mau phone_vn va long_digits co ty le duong tinh gia cao vi gia "
            "tien trong binh luan dien thoai thuong la day so dai.",
            "Ket qua la dieu kien CAN de trich vi du, khong phai dieu kien DU. "
            "Moi ban ghi bi gan co van phai duoc doc bang mat truoc khi dua vao "
            "luan van.",
        ],
        "per_split": per_split,
        "total_matches_by_pattern": dict(grand),
        "masked_examples": {
            k: [mask(x) for x in list(dict.fromkeys(v))[:args.limit]]
            for k, v in raw_examples.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    print()
    print(f"{'mau':<16}{'so khop':>9}   ghi chu do tin cay")
    print("-" * 78)
    for k, (_, note) in PATTERNS.items():
        print(f"{k:<16}{grand.get(k, 0):>9}   {note[:44]}")

    print()
    print("Vi du da che (khong phai ban goc):")
    for k, v in out["masked_examples"].items():
        if v:
            print(f"  {k:<16} {', '.join(v[:5])}")

    if args.show_raw:
        print()
        print("=== BAN GOC - chi in ra man hinh, khong ghi vao tep ===")
        for k, v in raw_examples.items():
            for x in list(dict.fromkeys(v))[:args.limit]:
                print(f"  {k:<16} {x}")

    print()
    print(f"Da luu -> {args.out}")


if __name__ == "__main__":
    main()
