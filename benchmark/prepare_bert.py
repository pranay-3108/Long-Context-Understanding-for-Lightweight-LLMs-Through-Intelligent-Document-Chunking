from __future__ import annotations

import argparse
from pathlib import Path

SIZES = [8000, 16000, 24000, 32000, 44000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the BERT paper PDF and make character-length benchmark stages.")
    parser.add_argument("pdf", help="Path to the BERT PDF")
    parser.add_argument("--out-dir", default="", help="Output folder; default is project papers/bert")
    args = parser.parse_args()

    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise SystemExit("PyPDF2 is required. Run: python -m pip install PyPDF2") from exc

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[1] / "papers" / "bert"
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)
    full_text = "\n\n".join(parts)
    (out_dir / "bert_full.txt").write_text(full_text, encoding="utf-8")
    print(f"Extracted chars: {len(full_text)}")

    for size in SIZES:
        content = full_text[:size]
        path = out_dir / f"bert_{size//1000}k.txt"
        path.write_text(content, encoding="utf-8")
        print(f"Saved {path.name}: {len(content)} chars")


if __name__ == "__main__":
    main()
