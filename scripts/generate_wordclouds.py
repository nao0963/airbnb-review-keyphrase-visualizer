"""
Airbnb Review Word Cloud Generator

This script generates word clouds from Airbnb review keyphrases and sentiment data.
It combines keyphrase extraction scores with sentiment polarity to create visualizations
that highlight positive sentiment phrases more prominently.

Usage:
    python generate_wordclouds.py sentiment.jsonl keyphrases.jsonl \
        --min_kp_score 0.9 --outdir out

Author: Portfolio Project
Description: Sentiment-weighted keyphrase visualization for Airbnb reviews
"""

import argparse
import json
import re
import math
import os
from collections import defaultdict
from pathlib import Path

# Optional wordcloud dependency check
WORDCLOUD_AVAILABLE = False
FONT_PATH = None
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
    # Search for common system fonts
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(font_path):
            FONT_PATH = font_path
            break
except Exception:
    WORDCLOUD_AVAILABLE = False

import matplotlib.pyplot as plt

# Configuration constants
ARTICLES = {"a", "an", "the"}
STOPWORDS = set()  # Add custom stopwords if needed

def load_jsonl(path):
    """
    Load and parse JSONL (JSON Lines) file.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        dict: Parsed JSON objects from each line
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def normalize_phrase(phrase: str) -> str:
    """
    Normalize keyphrase text for consistent processing.
    
    Removes leading/trailing punctuation, normalizes whitespace,
    removes common articles, and filters stopwords.
    
    Args:
        phrase: Raw keyphrase text
        
    Returns:
        str: Normalized and cleaned phrase
    """
    phrase = phrase.strip().lower()
    phrase = re.sub(r"^[^\w]+|[^\w]+$", "", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    tokens = phrase.split()
    if tokens and tokens[0] in ARTICLES:
        tokens = tokens[1:]
    tokens = [token for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)

def polarity_from_sentiment(sentiment_score, sentiment_label) -> float:
    """
    Calculate sentiment polarity score from 0 to 1.
    
    If sentiment scores are available, uses max(Positive - Negative, 0).
    Otherwise, falls back to binary label classification.
    
    Args:
        sentiment_score: Dictionary with Positive/Negative scores
        sentiment_label: String label (POSITIVE/NEGATIVE)
        
    Returns:
        float: Polarity score between 0 and 1
    """
    if sentiment_score and isinstance(sentiment_score, dict):
        positive = float(sentiment_score.get("Positive", 0.0))
        negative = float(sentiment_score.get("Negative", 0.0))
        return max(positive - negative, 0.0)
    if sentiment_label:
        label = str(sentiment_label).upper()
        return 1.0 if label == "POSITIVE" else 0.0
    return 0.0

def build_sentence_polarity_map(sentiment_jsonl_path):
    """
    Build a mapping of sentences to their sentiment polarity scores.
    
    Args:
        sentiment_jsonl_path: Path to sentiment analysis JSONL file
        
    Returns:
        dict: Mapping of (file_id, line_number) to polarity score
    """
    polarity_map = {}
    for record in load_jsonl(sentiment_jsonl_path):
        file_id = record.get("File")
        line_number = record.get("Line")
        if file_id is None or line_number is None:
            continue
        polarity = polarity_from_sentiment(
            record.get("SentimentScore"), 
            record.get("Sentiment")
        )
        polarity_map[(file_id, int(line_number))] = polarity
    return polarity_map

def build_weights(keyphrase_jsonl_path, polarity_map, min_keyphrase_score=0.9, drop_short_length=2):
    """
    Calculate word cloud weights using keyphrase scores and sentiment polarity.
    
    Word weight = Σ(KeyPhrase.Score × sentence_polarity) for each phrase.
    Filters out low-scoring keyphrases and very short phrases.
    
    Args:
        keyphrase_jsonl_path: Path to keyphrase extraction JSONL file
        polarity_map: Mapping of (file_id, line_number) to polarity scores
        min_keyphrase_score: Minimum keyphrase score threshold (default: 0.9)
        drop_short_length: Minimum phrase length to include (default: 2)
        
    Returns:
        dict: Mapping of normalized phrases to their computed weights
    """
    weights = defaultdict(float)
    for record in load_jsonl(keyphrase_jsonl_path):
        file_id = record.get("File")
        line_number = record.get("Line")
        
        # Get sentiment polarity for this sentence
        sentence_polarity = 0.0
        if file_id is not None and line_number is not None:
            sentence_polarity = float(polarity_map.get((file_id, int(line_number)), 0.0))
        
        # Process each keyphrase in the record
        for keyphrase in (record.get("KeyPhrases") or []):
            text = normalize_phrase(str(keyphrase.get("Text", "")))
            if not text:
                continue
            if drop_short_length and len(text) < drop_short_length:
                continue
            
            score = float(keyphrase.get("Score", 0.0))
            if score < min_keyphrase_score:
                continue
                
            weights[text] += score * sentence_polarity
    
    return dict(weights)

def save_wordcloud(weights: dict, title: str, output_file: Path):
    """
    Generate and save a word cloud visualization.
    
    If WordCloud library is available, creates a proper word cloud.
    Otherwise, falls back to a simple text grid layout.
    
    Args:
        weights: Dictionary mapping phrases to their weights
        title: Title for the visualization
        output_file: Path where to save the output image
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use WordCloud library if available
    if WORDCLOUD_AVAILABLE and len(weights) > 0:
        wordcloud = WordCloud(
            width=1200, 
            height=700, 
            background_color="white", 
            font_path=FONT_PATH
        )
        wordcloud.generate_from_frequencies(weights)
        
        plt.figure(figsize=(12, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close()
        return

    # Fallback: Simple text grid layout
    _save_text_grid_fallback(weights, title, output_file)


def _save_text_grid_fallback(weights: dict, title: str, output_file: Path):
    """
    Fallback visualization when WordCloud library is not available.
    Creates a simple grid layout of words sized by their weights.
    """
    plt.figure(figsize=(12, 7))
    plt.axis("off")
    plt.title(title)
    
    items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    if not items:
        plt.text(0.5, 0.5, "No data available", ha="center", va="center")
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close()
        return
    
    # Calculate grid dimensions
    num_items = len(items)
    columns = math.ceil(math.sqrt(num_items))
    rows = math.ceil(num_items / columns)
    
    # Scale font sizes based on weights
    max_weight = max(weight for _, weight in items)
    min_weight = min(weight for _, weight in items)
    
    def calculate_font_size(weight):
        if max_weight == min_weight:
            return 30
        return 10 + 50 * (weight - min_weight) / (max_weight - min_weight)
    
    # Place words in grid
    for index, (word, weight) in enumerate(items):
        row, col = divmod(index, columns)
        x = (col + 0.5) / columns
        y = 1 - (row + 0.5) / rows
        plt.text(x, y, word, fontsize=calculate_font_size(weight), 
                ha="center", va="center")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    """
    Main function to generate sentiment-weighted word clouds from Airbnb review data.
    """
    parser = argparse.ArgumentParser(
        description="Generate word clouds from Airbnb review keyphrases weighted by sentiment"
    )
    parser.add_argument("sentiment_jsonl", help="Path to sentiment analysis JSONL file")
    parser.add_argument("keyphrases_jsonl", help="Path to keyphrase extraction JSONL file")
    parser.add_argument("--min_keyphrase_score", "--min_kp_score", type=float, default=0.9,
                       help="Minimum keyphrase score threshold (default: 0.9)")
    parser.add_argument("--drop_short_length", "--drop_short_len", type=int, default=2,
                       help="Minimum phrase length to include (default: 2)")
    parser.add_argument("--output_dir", "--outdir", default="out",
                       help="Output directory for generated images (default: out)")
    
    args = parser.parse_args()

    # Build sentiment polarity mapping
    output_directory = Path(args.output_dir)
    polarity_map = build_sentence_polarity_map(args.sentiment_jsonl)
    
    # Calculate phrase weights using sentiment and keyphrase scores
    weights = build_weights(
        args.keyphrases_jsonl, 
        polarity_map,
        min_keyphrase_score=args.min_keyphrase_score,
        drop_short_length=args.drop_short_length
    )

    # Generate sentiment-weighted word cloud
    save_wordcloud(
        weights, 
        "Airbnb Review WordCloud - Frequency & Sentiment Weighted Keyphrases", 
        output_directory / "wordcloud.png"
    )

    # Display top phrases for verification
    top_phrases = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:20]
    print("Top 20 weighted phrases:")
    for phrase, weight in top_phrases:
        print(f"{phrase}\t{weight:.4f}")


if __name__ == "__main__":
    main()
