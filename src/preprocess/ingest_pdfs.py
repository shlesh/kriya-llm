import argparse, json, os, pathlib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from tqdm import tqdm

def extract_text_from_pdf(pdf_path: pathlib.Path, langs: str = "eng", dpi: int = 300):
    doc = fitz.open(pdf_path)
    pages_text = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        if len(text.strip()) < 50:  # likely scanned page -> OCR
            pix = page.get_pixmap(dpi=dpi, alpha=False, annots=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=langs)
        pages_text.append(text)
    return pages_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder with PDFs (recursive)")
    ap.add_argument("--out_dir", required=True, help="Output folder for JSONL files")
    ap.add_argument("--langs", default="eng+hin+san", help="Tesseract langs, e.g. 'eng+hin+san'")
    args = ap.parse_args()

    pdf_dir = pathlib.Path(args.pdf_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = [p for p in pdf_dir.rglob("*.pdf")]
    if not pdfs:
        print("No PDFs found.")
        return

    for pdf in tqdm(pdfs, desc="Extracting/OCR"):
        pages = extract_text_from_pdf(pdf, langs=args.langs)
        # write one JSONL per PDF (page-level entries)
        out_path = out_dir / (pdf.stem + ".jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for i, t in enumerate(pages):
                rec = {"doc": pdf.name, "page": i+1, "text": t}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
