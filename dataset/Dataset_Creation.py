import random
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import fitz
import os
import openai
import json
import csv
import pickle

# Read API Key
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

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

# Function to generate question_answer_pairs
def generate_question_answer_pairs(texts, n_pairs, temperature):
  """Generates a factoid question and answer from the given context."""

  print("Generating question and answering pairs...")

  qa_pairs = pd.DataFrame(columns=["context", "question", "answer"])


  for _ in range(n_pairs):

    context = random.choice(texts)

    prompt = f"""Assume the role of a student at HSLU with questions about modules, their content, and related information, seeking guidance from the RAG system.
      Your task is to write a factoid question and an answer given a context.
      Your factoid question should be answerable with a specific, concise piece of factual information from the context.
      Your factoid question should be formulated in the same style as questions users could ask in a search engine.
      This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

      Provide your answer as JSON data in the following format:

      Output:::
      {{
        \"question\": \"your factoid question\",
        \"answer\": \"your answer to the factoid question\"
      }}

      Now here is the context.

      Context: {context}\n
      Output:::"""

  
    try: # Generate a question-answer pair
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", 
             "content": prompt}
            ],
            max_tokens=300,
            temperature=temperature
            )
            
        output = json.loads(response.choices[0].message.content.strip())
        new_row = pd.DataFrame({"context": context,
                         "question": output['question'],
                         "answer": output['answer']}
                        , index=[0])
        
        qa_pairs = pd.concat([qa_pairs, new_row], ignore_index=True) 
        
    except Exception as e:
        print(f"Failover strategy activated due to: {e}")
        # Implement failover strategy here
        continue
  
  return qa_pairs

def evaluate_question(dataset, temperature):
    """Evaluates a question on groundedness, relevance, and clarity."""

    print("Evaluating question and answer pairs...")

    combined_prompt = """
    You will be given a question and a context (if applicable). Your task is to perform the following evaluations:

    **1. Question Groundedness**

    *  Can the question be answered unambiguously with the given context?
    * Rate on a scale of 1 to 5, where:
        * **1 (Not Answerable):** The question is too open-ended, relies on information not in the context, or is ambiguous. 
            * Example: "Is the Neural Networks module difficult?"
        * **3 (Somewhat Answerable):**  The question is answerable, but some details might depend on interpretation or additional assumptions.
        * **5 (Clearly Answerable):** The context provides all the necessary information for a single, unambiguous answer.
            * Example: "What is the workload in ECTS for the Machine Learning module?"

    *  Provide a rationale for your rating.

    **2. Question Relevance**

    *  How useful is this question to students at HSLU with questions about the corresponding modules, seeking guidance from the RAG system?
    * Rate on a scale of 1 to 5, where:
        * **1 (Not Useful):** The question is completely irrelevant to HSLU modules, asks about a non-existent module, or is too specific to be answered in a general guidance context. 
            * Example: "What is the WiFi password for the Lucerne campus?"
        * **3 (Somewhat Useful):** The question might be tangentially related to a module topic, but it's either too broad or too focused on a minor detail.
            * Example: "What year was the Recommender Systems module introduced?"
        * **5 (Extremely Useful):** The question is directly about a module's core content, or seeks general guidance that would benefit most HSLU students taking that module.
            * Example: "How do I find recommended readings for the Machine Learning module?"

    *  Provide a rationale for your rating.

    **3. Question Clarity**

    *  How well does the question stand on its own without needing additional context?
    *  Rate on a scale of 1 to 5, where 1 is "depends on additional context" and 5 is "makes sense by itself".
    *  Provide a rationale for your rating.

    **Output**

    Provide your answer as JSON data in the following format:

    Output:::
    {{
      \"eval_groundedness\": \"the rationale for the groundedness score\",
      \"groundedness_score\": \"your groundedness score 1 to 5\",
      \"eval_relevance\": \"the rationale for the relevance score\",
      \"relevance_score\": \"your relevance score 1 to 5\",
      \"eval_standalone\": \"the rationale for the standalone score\",
      \"standalone_score\": \"your standalone score 1 to 5\"
    }}

    Now here is the question and context.
    
    Question: {question}
    Context: {context} 
    Answer:::
    """

    score_columns = ["groundedness_score", "relevance_score", "standalone_score"]

    for index, row in dataset.iterrows():  # Iterate through DataFrame rows
        question = row['question']
        context = row['context']

        filled_prompt = combined_prompt.format(question=question, context=context)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system",
                           "content": filled_prompt}],
                max_tokens=500, 
                temperature=temperature
            )

            output = json.loads(response.choices[0].message.content.strip())

            # Update the DataFrame row with evaluation results
            dataset.loc[index, "eval_groundedness"] = output['eval_groundedness']
            dataset.loc[index, "groundedness_score"] = output['groundedness_score']
            dataset.loc[index, "eval_relevance"] = output['eval_relevance']
            dataset.loc[index, "relevance_score"] = output['relevance_score']
            dataset.loc[index, "eval_standalone"] = output['eval_standalone']
            dataset.loc[index, "standalone_score"] = output['standalone_score']

        except Exception as e:
            print(f"Failover strategy activated due to: {e}")
            # Implement failover strategy here
            continue

        # Calculate mean score  
        dataset.at[index, "mean_score"] = dataset.loc[index, score_columns].astype(float).mean(skipna=True)
   
    return dataset

def main():

    TEMPERATURE = 0.2
    N_PAIRS = 250

    # Directories setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_base_directory = os.path.join(script_dir, 'data')
    storage_dir = os.path.join(script_dir, 'embeddings')
    os.makedirs(storage_dir, exist_ok=True)

    topics_config = {
        "module_details": {
            "documents": [
                {
                    "filename": "module descriptions.pdf",
                    "embedders": {
                        "all-MiniLM-L6-v2": {"chunk_size": 200, "overlap": 40},
                        "bert-base-nli-mean-tokens": {"chunk_size": 100, "overlap": 20},
                        "roberta-large-nli-stsb-mean-tokens": {"chunk_size": 150, "overlap": 10},
                        "all-mpnet-base-v2": {"chunk_size": 200, "overlap": 50}
                    }
                }
            ]
        }
    }
    
    
    # Initialize API key
    api_key_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.txt')
    openai.api_key = read_api_key(api_key_file_path)

    # Set topic
    topic = "module_details"

    # Select available embedders based on the selected topic
    selected_embedder = "all-mpnet-base-v2"

    # Chunk the PDF
    pdf_directory = os.path.join(pdf_base_directory, topic)
    texts = load_and_chunk_texts(pdf_directory, topics_config[topic]["documents"], selected_embedder)

    # Initialize the SentenceTransformer model based on the selected embedder
    model = SentenceTransformer(selected_embedder)

    # Generate response
    response = generate_question_answer_pairs(texts, N_PAIRS, TEMPERATURE)

    # Evaluate questions
    eval_dataset = evaluate_question(response, TEMPERATURE)

    # Save the DataFrame to a CSV file
    eval_dataset.to_csv("pairs.csv", index=False, encoding='utf-8', sep='|', quoting=csv.QUOTE_MINIMAL, escapechar="\\")

    print("Generated saved to pairs.csv")

if __name__ == "__main__":
    main()
