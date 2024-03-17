from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import streamlit as st
import os
import openai

# Function to read the API key from a file
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

# Function to load embeddings and documents for a given topic
def load_embeddings_for_topic(topic):
    # Dynamically construct the paths to the embeddings and documents files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, 'embeddings', topic)
    embeddings_path = os.path.join(embeddings_dir, 'embeddings.npy')
    docs_path = os.path.join(embeddings_dir, 'docs.pkl')

    # Load the embeddings and documents
    embeddings = np.load(embeddings_path)
    docs = pd.read_pickle(docs_path)['document'].tolist()

    return embeddings, docs

# Initialize API key
api_key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
openai.api_key = read_api_key(api_key_file_path)

# Placeholder for the model (used only for encoding queries, if needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_documents(query_embedding, embeddings, docs, top_k):
    # Create a FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]

def generate_response_with_openai_chat(query, embeddings, docs, top_k, temperature, system_content):
    # Encode the query to get its embedding
    query_embedding = model.encode([query]).astype('float32')
    # Retrieve documents based on the query embedding
    retrieved_docs = retrieve_documents(query_embedding, embeddings, docs, top_k)
    # Form the context from retrieved documents
    context = ' '.join([doc[0] for doc in retrieved_docs])

    # Generate a response using OpenAI's API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": context}
        ],
        max_tokens=150,
        temperature=temperature
    )

    return response.choices[0].message['content'].strip()

st.title('RAG System Demo')

# Dynamically list available topics based on directory names
script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(script_dir, 'embeddings')
topics = [d for d in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, d))]
topic = st.selectbox('Select your topic of interest:', topics)

# Load the embeddings and documents for the selected topic
embeddings, docs = load_embeddings_for_topic(topic)

# Define default system contexts for each topic
default_system_contexts = {
    "module_details": "You are a helpful assistant specializing in Module Details. You should focus on providing details on the topic that the user is interested in."
                      "Hereby you focus on giving a complete picture: elaborate why a topic is important, for what purposes its used and what technologies or techniques can be used after learing about this topic."
                      "Please also judge the course difficulty on a scale of 1 to 5 where 1 is beginner, and 5 is professional.",
    "module_overview": "You are a knowledgeable guide for module overviews. Your job is to create bullet points on related modules while giving the most important information in a short format. Preferably bullet points."
                       "Hereby you list the ECTS, lecturers and module names. Group the modules either by lecturer, semester or subject of the topics when the user specifies his interests.",
    "module_planning": "Your expertise is in providing assistance in setting up schedules and semester planning. Use your knowledge and the context to provide a schedule like response that helps the user plan his studies."
}

# User settings for system context, top_k, and temperature
with st.expander("Set System Context"):
    # Fetch the default system context for the selected topic
    default_context = default_system_contexts.get(topic, "You are a helpful assistant.")
    user_defined_context = st.text_area("Enter the fixed system context you want the model to use:",
                                        default_context,
                                        help="This context helps guide the model's behavior.")
top_k = st.sidebar.number_input("Top K", min_value=1, max_value=10, value=5)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)

# Form submission for generating response
with st.form("my_form"):
    user_query = st.text_input("Enter your query:", help="Type your query here and press 'Generate Response'.")
    submitted = st.form_submit_button("Generate Response")
    if submitted and user_query:
        response = generate_response_with_openai_chat(user_query, embeddings, docs, top_k, temperature, user_defined_context)
        st.text_area("Response", response, height=300)
