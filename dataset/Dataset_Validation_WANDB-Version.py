from sentence_transformers import SentenceTransformer
import faiss
import os
import openai
import pickle
import pandas as pd
import csv
import json
import wandb

# WAND Library requires an account and the corresponding API Key from:
# Source: https://wandb.ai
# Over CLI, use "wandb login" to specify the API Key

# Function to read the API key from a file for OpenAI
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

# Function to load embeddings and documents for a given topic
def load_faiss_index_and_docs(topic, embedder_name, chunk, overlap):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, 'embeddings', topic)
    index_path = os.path.join(embeddings_dir, f'faiss_index_{embedder_name}_{chunk}-{overlap}.index')
    faiss_index = faiss.read_index(index_path)

    # Adjust the path for the documents file
    docs_path = os.path.join(embeddings_dir, f'docs_{embedder_name}_{chunk}-{overlap}.pkl')
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)

    return faiss_index, docs


def retrieve_documents(query_embedding, faiss_index, docs, top_k):
    # Search the FAISS index
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    return [(docs[i], distances[0][j]) for j, i in enumerate(indices[0])]


def format_input_for_model(query, retrieved_docs):
    # Join the content of the retrieved documents, each potentially with a tag
    document_contexts = '\n'.join([f"- {doc[0]} (Relevance Score: {doc[1]:.2f})" for doc in retrieved_docs])

    # Fill in the template
    query_context_template = f"""
    Question: {query}
    Relevant Information:
    {document_contexts}
    """
    formatted_input = query_context_template.format(query=query, document_contexts=document_contexts)
    return formatted_input

def generate_response(model, query, chunk, overlap, embeddings, docs, top_k, temperature, dataset, index):
    # Encode the query to get its embedding
    query_embedding = model.encode([query]).astype('float32')

    print("Current question:", query)
    # Retrieve documents based on the query embedding
    retrieved_docs = retrieve_documents(query_embedding, embeddings, docs, top_k)

    # Format the input for the model using the template
    formatted_input = format_input_for_model(query, retrieved_docs)

    reference_answer = dataset.loc[index, "answer"]

    # Prompt
    # Inspired and adapted by source: https://huggingface.co/learn/cookbook/en/rag_evaluation
    system_content = f"""
    **Overall Task:**

    ** Step 1: Generate an answer **
    Your task is to answer a question given a context.
    Based on the question ({query}) and relevant text ({formatted_input}),  carefully formulate an answer that you believe best captures the information. 

    
    ** Step 2: Evaluation Using Reference Answer**
    Based on the generated answer, compare it with the reference answer ({reference_answer}) and give a scoring from 1-5 based on the rubric below.
    Make sure to only give the score, so the numeric value itself, as an output and no other text.

    ### Score Rubric:
    [Focus on Correctness, Comprehensiveness, Clarity]
    Score 1: The generated answer is mostly wrong, misses important information, and/or is difficult to understand.
    Score 2: The generated answer has some inaccuracies, lacks some details from the reference, and/or has phrasing issues.
    Score 3: The generated answer gets the main point but could be more accurate, complete, or clearer.  
    Score 4: The generated answer aligns well with the reference answer and is easy to understand.
    Score 5: The generated answer is a near-perfect match to the reference answer in content and clarity.

    Provide your answer as JSON data in the following format:

    Output:::
        {{
          \"acc_score": \"Scoring given for the generated answer based on the reference answer\",
        }}

    Answer:::
    """
    
    # Include chat history in the messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": formatted_input}
    ]

    params = f"acc_{chunk}-{overlap}_k-{top_k}_t{temperature}"

    try:

        # Generate the response
        # Source: https://platform.openai.com/docs/api-reference/streaming?lang=python
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=temperature
        )

        output = json.loads(response.choices[0].message.content.strip())

        return(int(output['acc_score']))

    except Exception as e:
        print(f"Failover strategy activated due to: {e}")
        return 0


def train_and_evaluate():
    topic = "module_details"

    # Select available embedders based on the selected topic
    selected_embedder = "all-mpnet-base-v2"

    # Initialize the SentenceTransformer model dynamically based on the selected embedder
    model = SentenceTransformer(selected_embedder)

    # Read in dataset with questions/answers
    dataset = pd.read_csv("pairs.csv", sep=';', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar="\\")


    # Initialize a new wandb run
    with wandb.init() as run:
        # Access the hyperparameters through wandb.config
        chunk = run.config.chunk
        overlap = run.config.overlap
        k = run.config.top_k
        temp = run.config.temperature

        # Load the embeddings and docs to be used for response generation
        embeddings, docs = load_faiss_index_and_docs(topic, selected_embedder, chunk, overlap)

        # Counter
        i = 0

        # Total accuracy per combination
        acc_total = 0
        
        for index, row in dataset.iterrows():
            # Generate new answer for question and score it depending on the reference answer
            query = row['question']
            score = generate_response(model, query, chunk, overlap, embeddings, docs, k, temp, dataset, index)    
            i += 1

            # Add scoring accuracy to total accuracy
            acc_total += (score - 1) / 4
            
        acc_total = acc_total / i
        # Log metrics
        wandb.log({'acc_total': acc_total})

def main():
    # Initialize API key
    api_key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
    openai.api_key = read_api_key(api_key_file_path)

    # Specify sweep config
    # Source: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
    sweep_config = {
    'method': 'grid',  # Grid to check all combinations of parameters
    'metric': {
      'name': 'acc_total',  # Metric to maximize
      'goal': 'maximize'  # Operation
    },
    'parameters': {
        'chunk': {'values': [200, 400, 600]},
        'overlap': {'values': [50, 100]},
        'top_k': {'values': [2, 4, 6]},
        'temperature': {'values': [0.5, 1.25, 2]}
        }
    }

    # Start Sweeping Agent
    # Source: https://docs.wandb.ai/guides/sweeps/start-sweep-agents
    sweep_id = wandb.sweep(sweep_config, project="llm_eval")
    wandb.agent(sweep_id, train_and_evaluate)


if __name__ == "__main__":
    main()
