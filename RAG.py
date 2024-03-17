from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from transformers import pipeline
import streamlit as st
import os
import fitz  # PyMuPDF
import openai

# Function to read the API key from a file
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

api_key_file_path = 'config.txt'
openai.api_key = read_api_key(api_key_file_path)

# Load a pre-trained model for embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdf_texts(pdf_directory, chunk_size=500):
    """
    Load texts from PDFs and split into chunks.

    :param pdf_directory: Directory containing PDF files.
    :param chunk_size: Number of words in each chunk.
    """
    texts = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            with fitz.open(pdf_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                # Split text into chunks
                words = text.split()
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i+chunk_size])
                    texts.append(chunk)
    return texts

chunk_size = st.sidebar.number_input("Chunk Size", min_value=100, max_value=1000, value=500)

# Load texts from PDFs and Embedd
pdf_directory = 'C:/Users/daenu/Documents/Generative_AI_Project/data/'
docs = load_pdf_texts(pdf_directory, chunk_size)
doc_embeddings = model.encode(docs, show_progress_bar=True)

# Convert embeddings to FAISS compatible format
doc_embeddings = doc_embeddings.astype('float32')

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

def retrieve_documents(query, model, index, docs, top_k):
    query_embedding = model.encode([query])[0].astype('float32')
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]

def generate_response_with_openai_chat(query, top_k, temperature, system_content):
    retrieved_docs = retrieve_documents(query, model, index, docs, top_k)
    context = ' '.join([doc[0] for doc in retrieved_docs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},  # Use the user-defined system content
            {"role": "user", "content": context}
        ],
        max_tokens=150,
        temperature=temperature
    )

    generated_text = response.choices[0].message['content'].strip()
    return generated_text


st.title('RAG System Demo')

# Expandable section for setting fixed system content
with st.expander("Set System Context"):
    user_defined_context = st.text_area("Enter the fixed system context you want the model to use:",
                                        "You are a helpful assistant.",
                                        help="This context helps guide the model's behavior.")

top_k = st.sidebar.number_input("Top K", min_value=1, max_value=10, value=5)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)

with st.form("my_form"):
    user_query = st.text_input("Enter your query:", help="Type your query here and press 'Generate Response'.")
    submitted = st.form_submit_button("Generate Response")
    if submitted and user_query:
        response = generate_response_with_openai_chat(user_query, top_k, temperature, user_defined_context)
        st.text_area("Response", response, height=300)