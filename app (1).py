import pandas as pd
import streamlit as st
from groq import Groq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="Chatbot Kopi Senja", page_icon="☕")
st.title("☕ Chatbot Toko Kopi Senja")
st.caption("Tanya jam buka, harga, promo, ongkir, dan lainnya.")

# ===== 1. LOAD DATASET & VECTOR STORE (di-cache, jalan sekali saja) =====
@st.cache_resource(show_spinner="Menyiapkan chatbot...")
def load_vectorstore():
    # SESUAIKAN: nama file & kolom dengan dataset di notebook-mu
    df = pd.read_csv("dataset_toko_kopi_senja.csv")
    docs = [
        Document(
            page_content=(
                f"Kategori: {r['Kategori']} "
                f"Pertanyaan: {r['Pertanyaan']} "
                f"Jawaban: {r['Jawaban']}"
            )
        )
        for _, r in df.iterrows()
    ]
    # SESUAIKAN: pakai model embedding yang sama dengan di Colab
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return FAISS.from_documents(docs, emb)


@st.cache_resource
def load_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


vs = load_vectorstore()
client = load_client()

# SESUAIKAN: nama model Groq yang kamu pakai di Colab
MODEL = "llama-3.3-70b-versatile"

# ===== 2. FUNGSI TANYA (versi Streamlit dari tanya() di notebook) =====
def tanya(pertanyaan: str, k: int = 4):
    hasil = vs.similarity_search_with_relevance_scores(pertanyaan, k=k)
    konteks = "\n\n".join(d.page_content for d, _ in hasil)

    pesan = [
        {
            "role": "system",
            "content": (
                "Kamu adalah asisten ramah Toko Kopi Senja. "
                "Jawab HANYA berdasarkan konteks yang diberikan. "
                "Jika tidak ada di konteks, katakan bahwa kamu belum punya informasinya. "
                "Jawab singkat dalam Bahasa Indonesia."
            ),
        },
        {"role": "user", "content": f"KONTEKS:\n{konteks}\n\nPERTANYAAN: {pertanyaan}"},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=pesan, temperature=0.2
    )
    return resp.choices[0].message.content, hasil


# ===== 3. UI CHAT =====
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sumber"):
            with st.expander("📎 Sumber yang dipakai"):
                for i, (teks, skor) in enumerate(m["sumber"], 1):
                    st.write(f"[{i}] (skor {skor:.2f}) {teks[:200]}...")

if q := st.chat_input("Tulis pertanyaanmu..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            jawaban, hasil = tanya(q)
        st.markdown(jawaban)
        sumber = [(d.page_content.replace("\n", " "), s) for d, s in hasil]
        with st.expander("📎 Sumber yang dipakai"):
            for i, (teks, skor) in enumerate(sumber, 1):
                st.write(f"[{i}] (skor {skor:.2f}) {teks[:200]}...")

    st.session_state.messages.append(
        {"role": "assistant", "content": jawaban, "sumber": sumber}
    )
