This repository contains 2 major files
1. Create_Embeddings.py: This Script creates different embeddings based on the configurations defined
	in the Script. These embedders are stored locally such that they can be accessed later
2. RAG.py: This is a streamlit application that selects the before-mentioned locally stored embedders 
	and uses them to prompt a GPT3.5 Model given context fetched from the corresponding embedders.
	The app is a chat application that allows for customization in terms of Top-K and Temperature. 
	The chat history, as well as the passed Query are shown in separate Expandables.


To run the RAG.py App please proceed as follows:

1. Create a config.txt file in the project folder that contains the API Key for openAI. (This is not included in the git repository for Security reasons)
2. Create a venv where you install the python packages mentioned in the requirements.txt
3. Activate the venv and execute "streamlit run RAG.py"
4. The application should start on the localhost accessible via Browser.


Regarding the dataset subfolder:
1. Also requires a config.txt file containing an API key for OpenAI.
2. The Dataset-Creation / Embeddings-Creation should only be executed if the embedder has been modified.
3. A filtered dataset with question pairs ("pairs") already exists, and for validation, only one of the validation scripts needs to be used:
	- Excel-Version produces an output with the scores as a new CSV file.
	- WANDB-Version for generating a new sweep in the W&B Workspace (requires an API key)
