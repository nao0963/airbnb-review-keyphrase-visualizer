# Airbnb Review Keyphrase Visualizer

A data analysis project that extracts keyphrases and performs sentiment analysis on real business customer review data from 2023-2024, visualizing results through weighted word clouds.

## 📝 Project Overview

**Purpose**: Extract key phrases and sentiment information from customer review text to identify service strengths and improvement opportunities

**Data**: Airbnb review dataset (22 sample reviews in multiple languages)

**Methodology**:
- AWS Translate → Batch translation of multilingual source text files to English
- AWS Comprehend → Keyphrase extraction & Sentiment analysis (batch job)
- Python → Data preprocessing and weight calculation (frequency × score × sentiment)
- Python → WordCloud visualization

## 🔧 Technology Stack

- **Language**: Python 3.12
- **NLP Analysis**: AWS Comprehend (Natural Language Processing)
- **Visualization**: Matplotlib, WordCloud library
- **Data Processing**: JSON, JSONL format handling
- **Environment**: Virtual environment with pip requirements

## 📁 Project Structure

```
airbnb-review-keyphrase-visualizer/
├── data/                           # Processed analysis data
│   ├── keyphrases.jsonl           # Keyphrase extraction results
│   └── sentiment.jsonl            # Sentiment analysis results
├── image/                          # Process documentation images
│   ├── airbnb-review-web-raw.png  # Original data source
│   └── airbnb-review-keyphrase-extracted.png  # Extraction process
├── json/                           # AWS Comprehend output files
│   ├── output-keyphrase.json      # Raw keyphrase analysis
│   └── output-sentiment.json     # Raw sentiment analysis
├── result/                         # Generated visualizations
│   └── wordcloud.png             # Final weighted word cloud
├── scripts/                        # Analysis and visualization scripts
│   └── generate_wordclouds.py    # Main word cloud generation script
├── source/                         # Original dataset files
│   ├── airbnb-review-dataset-final.txt   # Original review text
│   ├── airbnb-review-dataset-final.csv   # Structured data format
│   └── airbnb-review-dataset-final.xlsx  # Spreadsheet format
├── translated/                     # Translation output
│   └── en.airbnb-review-dataset-final.txt  # English translated reviews
├── requirements.txt               # Python dependencies
└── venv/                          # Virtual environment
```

## 🌀 Analysis Process

### 1. Data Source
![Original Data](image/airbnb-review-web-raw.png)

Raw Airbnb reviews collected from actual business operations (2023-2024)

### 2. Text Extraction & Processing
![Keyphrase Extraction](image/airbnb-review-keyphrase-extracted.png)

- Multilingual review text extraction
- AWS Translate for consistent English processing
- AWS Comprehend batch processing for NLP analysis

### 3. Visualization Output
![WordCloud Result](result/wordcloud.png)

Sentiment-weighted keyphrase visualization highlighting positive customer feedback

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- AWS account with Comprehend and Translate access
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd airbnb-review-keyphrase-visualizer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Generate word cloud visualization:
```bash
python scripts/generate_wordclouds.py data/sentiment.jsonl data/keyphrases.jsonl --min_kp_score 0.9 --outdir result
```

**Parameters:**
- `--min_kp_score`: Minimum keyphrase confidence score (default: 0.9)
- `--drop_short_length`: Minimum phrase length to include (default: 2)
- `--outdir`: Output directory for generated images (default: out)

## 🔍 Analysis Insights

### Key Findings:
- **Overall Sentiment**: Reviews showed predominantly POSITIVE sentiment
- **Top Positive Keywords**: great place, very comfortable stay, subway station, host
- **Improvement Areas**: heating, trouble (mentioned in negative contexts)

### Customer Feedback Themes:
- **Location Convenience**: High praise for proximity to subway stations and amenities
- **Host Quality**: Consistent positive feedback about host friendliness and responsiveness  
- **Facility Issues**: Some mentions of heating problems requiring attention

## ✨ Technical Highlights

### Weight Calculation Algorithm
The visualization uses a sophisticated weighting system:
```
Word Weight = Keyphrase Score × Sentiment Polarity × Frequency
```

This approach ensures that:
- High-confidence keyphrases are emphasized
- Positive sentiment phrases appear more prominently
- Frequent positive mentions are highlighted

### Scalability
- Designed for small datasets but easily scalable to larger review collections
- Batch processing architecture supports high-volume analysis
- Modular design allows for easy integration with different data sources

## 📊 Data Processing Pipeline

1. **Input**: Raw multilingual review text
2. **Translation**: AWS Translate → Standardized English text
3. **NLP Analysis**: AWS Comprehend → Keyphrases + Sentiment scores
4. **Data Processing**: Python → Weight calculation and normalization
5. **Visualization**: WordCloud → Weighted visual representation

## 🛠️ Dependencies

Core requirements (see `requirements.txt`):
- `wordcloud>=1.9.0` - Word cloud generation
- `matplotlib>=3.5.0` - Plotting and visualization
- `numpy>=1.21.0` - Numerical computing
- `Pillow>=9.0.0` - Image processing support

## 📈 Business Value

This analysis framework provides actionable insights for:
- **Service Improvement**: Identify specific areas needing attention
- **Strength Recognition**: Understand what customers value most
- **Competitive Analysis**: Benchmark against customer expectations
- **Data-Driven Decisions**: Quantify qualitative feedback patterns

## 🔮 Future Enhancements

- **Trend Analysis**: Time-series sentiment tracking
- **Real-time Processing**: Live review analysis integration
- **Advanced NLP**: Custom entity recognition and topic modeling

---

*This project demonstrates the practical application of cloud-based NLP services for business intelligence and customer insight generation.*
