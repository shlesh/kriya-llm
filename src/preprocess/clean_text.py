import argparse, json, pathlib, re
from tqdm import tqdm
import sys, pathlib as _pl
sys.path.append(str(_pl.Path(__file__).resolve().parents[1]))
from preprocess.utils_text import normalize_text, remove_hyphenation, strip_headers_footers

def load_pages(jsonl_path: pathlib.Path):
    pages = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pages.append(obj["text"])
    return pages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder of JSONL (from ingest)")
    ap.add_argument("--out_dir", required=True, help="Folder for cleaned JSONL")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.jsonl"))
    if not files:
        print("No JSONL files found.")
        return

    for j in tqdm(files, desc="Cleaning"):
        pages = load_pages(j)
        pages = [normalize_text(p) for p in pages]
        pages = strip_headers_footers(pages, k=3)
        joined = "\n\n".join(pages)
        joined = remove_hyphenation(joined)
        # simple de-dup of empty lines
        joined = re.sub(r"\n{3,}", "\n\n", joined).strip()

        out_path = out_dir / j.name
        with out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"doc": j.stem, "text": joined}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
