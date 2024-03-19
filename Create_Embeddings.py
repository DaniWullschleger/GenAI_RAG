from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import fitz  # PyMuPDF
import time
import shutil


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
    return texts


def main():
    # List of embedders to use
    embedders = [
        ('all-MiniLM-L6-v2', 'all-MiniLM-L6-v2'),
        ('bert-base-nli-mean-tokens', 'bert-base-nli-mean-tokens'),
        ('roberta-large-nli-stsb-mean-tokens', 'roberta-large-nli-stsb-mean-tokens'),
        ('all-mpnet-base-v2', 'all-mpnet-base-v2')
    ]

    # Directories setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_base_directory = os.path.join(script_dir, 'data')
    storage_dir = os.path.join(script_dir, 'embeddings')
    os.makedirs(storage_dir, exist_ok=True)

    topics_config = {
        # Topics config remains the same
    }

    # Clear the topic-specific directory before storing new embeddings
    for topic, config in topics_config.items():
        topic_dir = os.path.join(storage_dir, topic)
        os.makedirs(topic_dir, exist_ok=True)
        clear_directory(topic_dir)

    for embedder_name, embedder_id in embedders:
        print(f"Processing with embedder: {embedder_name}")
        model = SentenceTransformer(embedder_id)  # Initialize the model for the current embedder

        for topic, config in topics_config.items():
            pdf_directory = os.path.join(pdf_base_directory, topic)
            os.makedirs(pdf_directory, exist_ok=True)  # Ensure PDF directory exists
            texts = load_and_chunk_texts(pdf_directory, config["documents"])  # Load and chunk texts

            start_time = time.time()  # Capture start time
            docs_embeddings = model.encode(texts, show_progress_bar=True)  # Generate embeddings
            end_time = time.time()  # Capture end time
            duration = end_time - start_time  # Calculate duration

            docs_embeddings_np = np.array(docs_embeddings)
            topic_dir = os.path.join(storage_dir, topic)
            os.makedirs(topic_dir, exist_ok=True)

            # Adjust filenames to include the embedder's name
            embeddings_filename = f'embeddings_{embedder_name}.npy'
            docs_filename = f'docs_{embedder_name}.pkl'

            embeddings_path = os.path.join(topic_dir, embeddings_filename)
            docs_path = os.path.join(topic_dir, docs_filename)

            # Store embeddings and documents
            np.save(embeddings_path, docs_embeddings_np)
            pd.DataFrame(texts, columns=['document']).to_pickle(docs_path)

            print(f"Processed and saved embeddings and documents for {topic} with {embedder_name}")
            print(f"Embedding generation for {topic} using {embedder_name} took {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
