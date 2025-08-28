import argparse, json, pathlib, random
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder of cleaned JSONL (one line per doc)")
    ap.add_argument("--out_dir", required=True, help="Output dir for train/val jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for j in sorted(in_dir.glob("*.jsonl")):
        with j.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if len(obj.get("text","").strip()) > 0:
                    docs.append(obj)

    random.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_ratio))
    val = docs[:n_val]
    train = docs[n_val:]

    with (out_dir/"train.jsonl").open("w", encoding="utf-8") as f:
        for d in train:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with (out_dir/"val.jsonl").open("w", encoding="utf-8") as f:
        for d in val:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train and {len(val)} val docs.")

if __name__ == "__main__":
    main()
