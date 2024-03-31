from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
import os
import openai
import re
import pickle
from datetime import datetime

# Read the API key from a file (ignored on git)
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()


# Dynamically list available embedders for the selected topic in the streamlit dropdown
def list_available_embedders(topic):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, 'embeddings', topic)

    # Regex expressions were done using ChatGPT
    embeddings_files = [f for f in os.listdir(embeddings_dir) if re.match(r'faiss_index_.*\.index', f)]
    embedder_names = [re.sub(r'faiss_index_(.*)\.index', r'\1', f) for f in embeddings_files]
    return embedder_names


# Load embeddings and documents for a given topic
def load_faiss_index_and_docs(topic, embedder_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, 'embeddings', topic)
    index_path = os.path.join(embeddings_dir, f'faiss_index_{embedder_name}.index')
    faiss_index = faiss.read_index(index_path)

    # Adjust the path for the documents file
    docs_path = os.path.join(embeddings_dir, f'docs_{embedder_name}.pkl')
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)

    return faiss_index, docs


# Search the FAISS index for matching documents
def retrieve_documents(query_embedding, faiss_index, docs, top_k):
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    return [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]


# Format the request, such that also the topic and some meta-data about the document are included.
# Basic usage of a prompt template.
def format_input_for_model(topic, query, retrieved_docs, ):
    document_contexts = '\n'.join([f"- {doc[0]} (Relevance Score: {doc[1]:.2f})" for doc in retrieved_docs])

    query_context_template = f"""Topic: {topic}
    Question: {query}
    Relevant Information:
    {document_contexts}
    """

    formatted_input = query_context_template.format(topic=topic, query=query, document_contexts=document_contexts)
    return formatted_input


def generate_response(model, query, embeddings, docs, top_k, temperature, system_content, topic):
    # Encode the query to get its embedding
    query_embedding = model.encode([query]).astype('float32')

    # Search the "nearest neighbors" of the query in the embedding space
    retrieved_docs = retrieve_documents(query_embedding, embeddings, docs, top_k)

    # Format the input for the model using a basic template
    formatted_input = format_input_for_model(topic, query, retrieved_docs)


    # Store the latest query and chat history in session state variables of streamlit (since streamlit runs the
    # whole code at each interaction)
    st.session_state['latest_query'] = formatted_input
    messages = st.session_state['chat_history'] + [
        {"role": "system", "content": system_content},
        {"role": "user", "content": formatted_input}
    ]

    # Do the API call to get the response. Handle cases where the API key fails.
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300,
            temperature=temperature
        )
        response_text = response.choices[0].message['content'].strip()
    except:
        response_text = "Error during API Call"



    # Add latest query and resonse including time stamp to the chat history.
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state['chat_history'].append({"role": "user", "content": query})
    st.session_state['timestamps'].append(current_time)
    st.session_state['chat_history'].append({"role": "assistant", "content": response_text})
    st.session_state['timestamps'].append(current_time)

    return response_text


def main():
    # Fetch the locally stored API key
    api_key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
    openai.api_key = read_api_key(api_key_file_path)

    st.title('RAG System Demo')

    # Only initialize the chat history if none exists yet.
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'timestamps' not in st.session_state:
        st.session_state['timestamps'] = []
    if 'latest_query' not in st.session_state:
        st.session_state['latest_query'] = ""

    # Dynamically list available topics based on the directories created in Create_Embeddings.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, 'embeddings')
    topics = [d for d in os.listdir(embeddings_dir) if os.path.isdir(os.path.join(embeddings_dir, d))]
    topic = st.selectbox('Select your topic of interest:', topics)

    # List available embedders based on the selected topic
    embedder_names = list_available_embedders(topic)
    selected_embedder = st.selectbox('Select an embedder:', embedder_names)

    # Initialize the SentenceTransformer model dynamically based on the selected embedder
    model_identifier = selected_embedder
    model = SentenceTransformer(model_identifier)

    # Load the embeddings and documents for the selected topic and embedder
    embeddings, docs = load_faiss_index_and_docs(topic, selected_embedder)

    # Basic system contexts to provide the model with some topic-specific instructions
    default_system_contexts = {
        "module_details": "You are a helpful assistant specializing in Module Details. You should focus on providing details on the topic that the user is interested in."
                          "Hereby you focus on giving a complete picture: elaborate why a topic is important, for what purposes its used and what technologies or techniques can be used after learing about this topic."
                          "Please also judge the course difficulty on a scale of 1 to 5 where 1 is beginner, and 5 is professional.",
        "module_overview": "You are a knowledgeable guide for module overviews. Your job is to create bullet points on related modules while giving the most important information in a short format. Preferably bullet points."
                           "Hereby you list the ECTS, lecturers and module names. Group the modules either by lecturer, semester or subject of the topics when the user specifies his interests."
    }

    # Adjust the system context based on the selected topic.
    with st.expander("Set System Context"):
        default_context = default_system_contexts.get(topic)
        system_context = st.text_area("Enter the fixed system context you want the model to use:", default_context)

    # Sidebar for user settings on top_k, temperature and clearing chat history
    top_k = st.sidebar.number_input("Top K", min_value=1, max_value=10, value=5)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)

    # Handle "Clear Chat History" button-click by re-initializing the session state.
    if st.sidebar.button('Clear Chat History'):
        st.session_state['chat_history'] = []
        st.session_state['timestamps'] = []
        st.session_state['latest_query'] = ""
        st.sidebar.success('Chat history cleared!')

    # Text field and button for submitting a query
    user_query = st.text_input("Enter your query:", help="Type your query here and press 'Generate Response'.")
    submitted = st.button("Generate Response")
    if submitted and user_query:
        response = generate_response(model, user_query, embeddings, docs, top_k, temperature, system_context, topic)
        st.text_area("Response", response, height=300)

    # Expander to display the latest query
    query_expander = st.expander("Latest Query to Model", expanded=False)
    with query_expander:
        st.text(st.session_state['latest_query'])

    # Expander to display chat history in reverse order (New on top, Old below). Code Snippet created using ChatGPT
    chat_expander = st.expander("Show Chat History", expanded=False)
    with chat_expander:
        # Use slicing to get pairs of (user, assistant) messages for display
        for i in range(len(st.session_state['chat_history']) - 2, -1, -2):
            user_message = st.session_state['chat_history'][i]
            assistant_message = st.session_state['chat_history'][i + 1]
            user_timestamp = st.session_state['timestamps'][i]
            assistant_timestamp = st.session_state['timestamps'][i + 1]

            # Display user message
            if user_message['role'] == 'user':
                chat_expander.markdown(f"**You:** at **{user_timestamp}** \n{user_message['content']} ")

            # Display assistant message
            if i + 1 < len(st.session_state['chat_history']):
                if assistant_message['role'] == 'assistant':
                    chat_expander.markdown(
                        f"**Assistant:** at **{assistant_timestamp}** \n{assistant_message['content']}")

            # Visually separate conversations except after last message
            if i > 0:
                chat_expander.markdown("---")


if __name__ == "__main__":
    main()
