from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import fitz  # PyMuPDF

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_and_chunk_texts(pdf_directory, documents):
    texts = []
    for doc in documents:
        filename = doc["filename"]
        chunk_size = doc["chunk_size"]
        pdf_path = os.path.join(pdf_directory, filename)
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text()
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i + chunk_size])
                        texts.append(chunk)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        print(f"Processed {filename}: {len(texts)} chunks")
    return texts


# Base directory for PDFs
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_base_directory = os.path.join(script_dir, 'data')

topics_config = {
    "module_overview": {
        "documents": [
            {"filename": "module descriptions.pdf", "chunk_size": 200},
            {"filename": "module table.pdf", "chunk_size": 50},
        ]
    },
    "module_details": {
        "documents": [
            {"filename": "module descriptions.pdf", "chunk_size": 200}
        ]
    },
    "module_planning": {
        "documents": [
            {"filename": "timetable.pdf", "chunk_size": 50},
            {"filename": "general_info_msc.pdf", "chunk_size": 100},
            {"filename": "plan-default.pdf", "chunk_size": 100},
        ]
    },
}

# Directory to store the embeddings and documents
storage_dir = os.path.join(script_dir, 'embeddings')
os.makedirs(storage_dir, exist_ok=True)

for topic, config in topics_config.items():
    # Determine the full path to the PDFs for the current topic
    pdf_directory = os.path.join(pdf_base_directory, topic)

    # Ensure the PDF directory exists
    os.makedirs(pdf_directory, exist_ok=True)

    # Load and chunk texts for the current topic
    texts = load_and_chunk_texts(pdf_directory, config["documents"])

    # Generate embeddings
    docs_embeddings = model.encode(texts, show_progress_bar=True)
    docs_embeddings_np = np.array(docs_embeddings)

    # Create topic-specific directory within the storage directory
    topic_dir = os.path.join(storage_dir, topic)
    os.makedirs(topic_dir, exist_ok=True)

    # Store embeddings and documents
    embeddings_path = os.path.join(topic_dir, 'embeddings.npy')
    docs_path = os.path.join(topic_dir, 'docs.pkl')

    np.save(embeddings_path, docs_embeddings_np)
    pd.DataFrame(texts, columns=['document']).to_pickle(docs_path)

    print(f"Processed and saved embeddings and documents for {topic}")
