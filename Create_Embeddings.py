from sentence_transformers import SentenceTransformer
import numpy as np
import os
import fitz  # PyMuPDF
import time
import shutil
import faiss
import pickle


def clear_directory(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


def load_and_chunk_texts(pdf_directory, documents, embedder_id):
    texts = []
    for doc in documents:
        filename = doc["filename"]
        if embedder_id not in doc["embedders"]:
            print(f"Embedder {embedder_id} configuration not found for {filename}. Skipping.")
            continue
        chunk_size = doc["embedders"][embedder_id]["chunk_size"]
        overlap = doc["embedders"][embedder_id]["overlap"]
        pdf_path = os.path.join(pdf_directory, filename)
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text = page.get_text()
                    words = text.split()
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = ' '.join(words[i:i + chunk_size])
                        texts.append(chunk)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return texts


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use IndexFlatL2 for simplicity; consider other indexes for larger datasets
    index.add(embeddings)
    return index

def main():
    embedders = [
        'all-MiniLM-L6-v2',
        'bert-base-nli-mean-tokens',
        'roberta-large-nli-stsb-mean-tokens',
        'all-mpnet-base-v2'
    ]

    # Directories setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_base_directory = os.path.join(script_dir, 'data')
    storage_dir = os.path.join(script_dir, 'embeddings')
    os.makedirs(storage_dir, exist_ok=True)

    topics_config = {
        "module_overview": {
            "documents": [
                {
                    "filename": "module_requirements_table.txt",
                    "embedders": {
                        "all-MiniLM-L6-v2": {"chunk_size": 30, "overlap": 0},
                        "bert-base-nli-mean-tokens": {"chunk_size": 50, "overlap": 0},
                        "roberta-large-nli-stsb-mean-tokens": {"chunk_size": 50, "overlap": 0},
                        "all-mpnet-base-v2": {"chunk_size": 50, "overlap": 0}
                    }
                },
                {
                    "filename": "module table.pdf",
                    "embedders": {
                        "all-MiniLM-L6-v2": {"chunk_size": 30, "overlap": 0},
                        "bert-base-nli-mean-tokens": {"chunk_size": 40, "overlap": 5},
                        "roberta-large-nli-stsb-mean-tokens": {"chunk_size": 50, "overlap": 10},
                        "all-mpnet-base-v2": {"chunk_size": 50, "overlap": 10}
                    }
                }
            ]
        },
        "module_details": {
            "documents": [
                {
                    "filename": "module descriptions.pdf",
                    "embedders": {
                        "all-MiniLM-L6-v2": {"chunk_size": 200, "overlap": 40},
                        "bert-base-nli-mean-tokens": {"chunk_size": 100, "overlap": 20},
                        "roberta-large-nli-stsb-mean-tokens": {"chunk_size": 150, "overlap": 10},
                        "all-mpnet-base-v2": {"chunk_size": 250, "overlap": 30}
                    }
                }
            ]
        }
    }

    # Clear the topic-specific directory before storing new embeddings
    for topic, config in topics_config.items():
        topic_dir = os.path.join(storage_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        clear_directory(topic_dir)

    for embedder_id in embedders:
        print(f"Processing with embedder: {embedder_id}")
        model = SentenceTransformer(embedder_id)  # Initialize the model for the current embedder

        for topic, config in topics_config.items():
            topic_dir = os.path.join(storage_dir, topic)
            pdf_directory = os.path.join(pdf_base_directory, topic)
            os.makedirs(pdf_directory, exist_ok=True)
            texts = load_and_chunk_texts(pdf_directory, config["documents"], embedder_id)

            start_time = time.time()
            docs_embeddings = model.encode(texts, show_progress_bar=True)  # Generate embeddings
            end_time = time.time()
            duration = end_time - start_time

            # Convert embeddings to NumPy array for FAISS
            docs_embeddings_np = np.array(docs_embeddings).astype('float32')
            # Create and store FAISS index
            faiss_index = create_faiss_index(docs_embeddings_np)
            index_path = os.path.join(topic_dir, f'faiss_index_{embedder_id}.index')
            faiss.write_index(faiss_index, index_path)

            # Store documents
            docs_path = os.path.join(topic_dir, f'docs_{embedder_id}.pkl')
            with open(docs_path, 'wb') as f:
                pickle.dump(texts, f)

            print(f"Processed and saved embeddings, documents, and FAISS index for {topic} with {embedder_id}")
            print(f"Embedding generation for {topic} using {embedder_id} took {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
