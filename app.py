import arxiv
import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )
    return embedding_model, vectorizer

@st.cache_data(show_spinner="Fetching papers...")
def fetch_papers(query, max_results=100):
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3,
        num_retries=2
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    try:
        for result in client.results(search):
            papers.append({
                "title": result.title,
                "abstract": result.summary.replace("\n", " "),
                "url": result.entry_id,
                "date": result.published.strftime("%Y-%m-%d"),
                "authors": ", ".join(a.name for a in result.authors[:3])
            })
    except Exception as e:
        if "429" in str(e):
            st.error("arXiv is rate limiting us. Please wait 5 minutes and try again.")
            st.stop()
        else:
            raise e
    return papers

@st.cache_data(show_spinner="Modeling topics...")
def run_bertopic(query, max_results):
    papers = fetch_papers(query, max_results)
    docs = [f"{p['title']}. {p['abstract'][:300]}" for p in papers]
    embedding_model, vectorizer = load_models()
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        hdbscan_model=KMeans(n_clusters=8),
        verbose=False
    )
    topics, _ = topic_model.fit_transform(docs)
    return papers, topics, topic_model

st.set_page_config(page_title="Paper Monitor", layout="wide")
st.title("Research trend monitor")
st.caption("Discover emerging topics in academic research — powered by arXiv + BERTopic")

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input(
        "Search research field",
        placeholder="e.g. computer vision, climate change, drug discovery..."
    )
with col2:
    max_results = st.slider("Papers to analyze", 50, 200, 100, step=50)

if not query:
    st.info("Enter a research field and click Run analysis.")
    st.stop()

if st.button("Run analysis", type="primary"):
    papers, topics, topic_model = run_bertopic(query, max_results)

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].head(10)

    st.subheader("Topic landscape")
    chart_data = pd.DataFrame({
        "Topic": [" · ".join([w for w, _ in topic_model.get_topic(t)[:3]]) for t in topic_info["Topic"]],
        "Papers": topic_info["Count"].values
    })
    st.bar_chart(chart_data.set_index("Topic"))

    st.subheader("Explore topics")
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        keywords = " · ".join([w for w, _ in topic_model.get_topic(topic_id)[:5]])
        topic_papers = [papers[i] for i, t in enumerate(topics) if t == topic_id]

        with st.expander(f"{keywords}  —  {row['Count']} papers"):
            for p in topic_papers[:5]:
                st.markdown(f"**[{p['title']}]({p['url']})**")
                st.caption(f"{p['date']}  ·  {p['authors']}")
                st.write(p['abstract'][:300] + "...")
                st.divider()