# Research trend monitor

Discover emerging topics in academic research powered by arXiv and BERTopic.

Search any research field and get an instant map of what's being studied right now — clustered into meaningful topics with representative papers for each.

## What it does

- Fetches the latest papers from arXiv for any research field you type
- Groups them into topic clusters using BERTopic and sentence embeddings
- Shows a bar chart of the topic landscape
- Lets you explore papers per topic with titles, authors, dates and abstracts
- Caches results for 24 hours so repeated searches are instant

## Tech stack

- [Streamlit](https://streamlit.io) — web interface
- [arXiv API](https://arxiv.org/help/api) — paper data, free and official
- [BERTopic](https://maartengr.github.io/BERTopic) — topic modeling
- [sentence-transformers](https://www.sbert.net) — text embeddings
- [scikit-learn](https://scikit-learn.org) — KMeans clustering

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/YOURNAME/paper-monitor.git
cd paper-monitor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Usage

1. Type a research field in the search box — e.g. `computer vision`, `drug discovery`, `climate change`
2. Choose how many papers to analyze (50–200)
3. Click **Run analysis**
4. Explore the topic landscape and click any topic to see representative papers

## Notes

- arXiv rate limits requests to ~1 per 3 seconds. If you see a 429 error, wait 5 minutes and try again
- Results are cached for 24 hours per query — repeated searches within the same day load instantly
- First run downloads the embedding model (~80MB), cached after that

## Roadmap

- [ ] Supabase integration for persistent cross-session caching
- [ ] Weekly email digest of trending topics
- [ ] Compare topics across two fields side by side
- [ ] Rising vs established topic detection across weeks

## License

MIT