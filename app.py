import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Settings ---
CSV_PATH = "faq_clean.csv"  # CSV file must be in same repo
Q_COL = "body"              # column with question text
A_COL = "answer"            # column with answer text
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 Simple FAQ Chatbot")

# Load data
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error(f"CSV file '{CSV_PATH}' not found. Please check file name and path.")
    st.stop()

st.caption(f"Loaded {len(df)} FAQs")

# Cache model and embeddings
@st.cache_resource
def get_model(name):
    return SentenceTransformer(name)

@st.cache_data
def encode_questions(texts):
    texts = ["" if pd.isna(t) else str(t) for t in texts]
    embs = get_model(MODEL_NAME).encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype="float32")

q_embs = encode_questions(df[Q_COL].astype(str).tolist())

# User interface
user_q = st.text_input("Ask your question")
top_k = st.slider("Number of suggestions", min_value=1, max_value=5, value=1)
threshold = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.01)

if user_q:
    qv = get_model(MODEL_NAME).encode([user_q], normalize_embeddings=True).astype("float32")
    scores = (q_embs @ qv.T).squeeze()  # cosine similarity (dot on normalized vectors)

    # sort by score
    idxs = np.argsort(-scores)
    shown = 0
    for idx in idxs:
        if scores[idx] < threshold:
            continue
        question = df.iloc[idx][Q_COL]
        answer = df.iloc[idx][A_COL]
        st.markdown(f"### 🟢 Answer (score: {scores[idx]:.3f})")
        st.write(answer)
        with st.expander("Matched question"):
            st.write(question)
        shown += 1
        if shown >= top_k:
            break

    if shown == 0:
        st.warning("No match above the threshold. Try lowering the threshold or rephrasing your question.")
