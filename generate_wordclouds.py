# generate_wordclouds.py
# Usage:
#   python generate_wordclouds.py sentiment.jsonl keyphrases.jsonl \
#       --min_kp_score 0.9 --outdir out

import argparse, json, re, math, os
from collections import defaultdict
from pathlib import Path

# Optional deps
WORDCLOUD_AVAILABLE = False
FONT_PATH = None
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
    # Try common TTF
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(fp):
            FONT_PATH = fp
            break
except Exception:
    WORDCLOUD_AVAILABLE = False

import matplotlib.pyplot as plt

ARTICLES = {"a", "an", "the"}
STOPWORDS = set()  # 필요시 추가

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    toks = s.split()
    if toks and toks[0] in ARTICLES:
        toks = toks[1:]
    toks = [t for t in toks if t not in STOPWORDS]
    return " ".join(toks)

def polarity_from_sentiment(sent_score, sent_label) -> float:
    """0~1. 점수 있으면 max(Positive - Negative, 0), 없으면 라벨로 대체."""
    if sent_score and isinstance(sent_score, dict):
        pos = float(sent_score.get("Positive", 0.0))
        neg = float(sent_score.get("Negative", 0.0))
        return max(pos - neg, 0.0)
    if sent_label:
        lab = str(sent_label).upper()
        return 1.0 if lab == "POSITIVE" else 0.0
    return 0.0

def build_sentence_polarity_map(sent_jsonl_path):
    pol_map = {}
    for rec in load_jsonl(sent_jsonl_path):
        file_id = rec.get("File")
        line_no = rec.get("Line")
        if file_id is None or line_no is None:
            continue
        pol = polarity_from_sentiment(rec.get("SentimentScore"), rec.get("Sentiment"))
        pol_map[(file_id, int(line_no))] = pol
    return pol_map

def build_weights(keyphr_jsonl_path, pol_map, min_kp_score=0.9, drop_short_len=2):
    """워드클라우드 가중치 = Σ(KeyPhrase.Score × sentence_polarity)."""
    weights = defaultdict(float)
    for rec in load_jsonl(keyphr_jsonl_path):
        file_id = rec.get("File")
        line_no = rec.get("Line")
        sent_pol = float(pol_map.get((file_id, int(line_no))) if (file_id is not None and line_no is not None) else 0.0)
        for kp in (rec.get("KeyPhrases") or []):
            text = normalize_phrase(str(kp.get("Text", "")))
            if not text:
                continue
            if drop_short_len and len(text) < drop_short_len:
                continue
            s = float(kp.get("Score", 0.0))
            if s < min_kp_score:
                continue
            weights[text] += s * sent_pol
    return dict(weights)

def save_wordcloud(weights: dict, title: str, outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if WORDCLOUD_AVAILABLE and len(weights) > 0:
        wc = WordCloud(width=1200, height=700, background_color="white", font_path=FONT_PATH)
        wc.generate_from_frequencies(weights)
        plt.figure(figsize=(12,7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close()
        return

    # Fallback: 간단한 텍스트 배치
    plt.figure(figsize=(12,7))
    plt.axis("off")
    plt.title(title)
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    if not items:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.savefig(outfile, dpi=200, bbox_inches="tight"); plt.close(); return
    n = len(items); cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols)
    max_w = max(w for _, w in items); min_w = min(w for _, w in items)
    def scale(w):
        if max_w == min_w: return 30
        return 10 + 50 * (w - min_w) / (max_w - min_w)
    for idx, (word, w) in enumerate(items):
        r, c = divmod(idx, cols)
        x = (c + 0.5) / cols; y = 1 - (r + 0.5) / rows
        plt.text(x, y, word, fontsize=scale(w), ha="center", va="center")
    plt.tight_layout(); plt.savefig(outfile, dpi=200, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentiment_jsonl")
    ap.add_argument("keyphrases_jsonl")
    ap.add_argument("--min_kp_score", type=float, default=0.9)
    ap.add_argument("--drop_short_len", type=int, default=2)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    pol_map = build_sentence_polarity_map(args.sentiment_jsonl)
    weights = build_weights(args.keyphrases_jsonl, pol_map,
                            min_kp_score=args.min_kp_score,
                            drop_short_len=args.drop_short_len)

    # 전체(양의 극성 반영) 가중치 워드클라우드
    save_wordcloud(weights, "Word Cloud — Frequency × Score × Sentiment(+)", outdir / "wordcloud_hybrid.png")

    # 필요하면 상위 키 확인
    top = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
    print("Top 20:")
    for k, v in top:
        print(f"{k}\t{v:.4f}")

if __name__ == "__main__":
    main()
