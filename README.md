# Semantic Chunking and Finetuned LLM Adaptation Strategy for RAG-Chatbots in Healthcare 5.0 IR

This project aims to enhance medical information retrieval for remote patients in the era of Healthcare 5.0. It leverages Language Model (LLM) finetuning and Semantic Chunking within a Retrieval-Augmented Generation (RAG) based Chatbot framework to provide personalized information from various medical documents, addressing the challenge of LLM hallucinations.

## Features

1. **Localized Setup**: The entire setup, including the LLM and the database, runs locally on the CPU.
2. **Qdrant Vector Database**: The project utilizes a Qdrant Docker database for storing chunks of medical documents.
3. **Semantic Chunking**: Semantic chunking techniques are employed to derive chunks from the provided medical documents.
4. **Finetuned LLM**: The project uses the BioMistral-7B model, a MEDICAL DOMAIN FINE-tuned model, for enhanced understanding and learning of medical information.
5. **Word Embeddings**: Pubmed-bert is used for obtaining dense embeddings of words.

## Getting Started

### Prerequisites

- Docker (for running the Qdrant database)
- Python 3.x
- Required Python packages (specified in the `requirements.txt` file)

### Installation

1. Clone the repository: git clone [https://github.com/your-repo/semantic-chunking-rag-chatbot.git](https://github.com/fenil210/Medical-RAG)

2. Install the required Python packages: pip install -r requirements.txt

3. Download the BioMistral-7B.Q4_K_M.gguf model from [Hugging Face](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF) and place it in the project directory.

### Running the Docker Database

#### Windows

1. Download Docker Desktop.
2. Open PowerShell.
3. Run `docker pull qdrant/qdrant` to pull the Qdrant Docker image.
4. Run `docker images` to check the available images.
5. Run `docker ls` to check the running containers.
6. Run `docker run -p 6333:6333 qdrant/qdrant` to start the Qdrant container.
![image](https://github.com/fenil210/Medical-RAG/assets/121050723/6579c70e-e697-40b2-8142-72ea175ae1a0)


### Usage

1. Run `python ingest.py` to create the database and ingest the medical documents. The Qdrant dashboard will be available at `http://localhost:6333/dashboard`.
![image](https://github.com/fenil210/Medical-RAG/assets/121050723/f2235852-0153-4db8-991f-57cfb9714641)

![image](https://github.com/fenil210/Medical-RAG/assets/121050723/9d7fa324-db51-47b7-bde3-0a9eb49aa04b)

![image](https://github.com/fenil210/Medical-RAG/assets/121050723/1079fad4-baeb-402b-86ae-47e412da00eb)



2. Run `python retriever.py` to check if the model is responding well.

Top 2 retrived chunks with meta-data based on question : What is Metastatic disease?
![image](https://github.com/fenil210/Medical-RAG/assets/121050723/3bee2ee8-4a8e-48e6-8b68-fcb4d4dcc858)


3. Run `uvicorn rag:app` to start the FastAPI and Flask-based application. (check at:  http://127.0.0.1:8000)
- the app will give output on an average 25-30 seconds due to LLM running on local machine with CPU. It will also give the context along with meta-data such as from which document, from which page etc. 
![image](https://github.com/fenil210/Medical-RAG/assets/121050723/e855e0fe-771a-4720-a070-6fd2eaccfd14)

![image](https://github.com/fenil210/Medical-RAG/assets/121050723/6dbf6c87-e9ae-4042-a7b3-4550b9a3f5af)


## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Create a pull request, describing your changes in detail.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the vector database.
- [Hugging Face](https://huggingface.co/) for the BioMistral-7B model and Pubmed-bert.
- [FastAPI](https://fastapi.tiangolo.com/) and [Flask](https://flask.palletsprojects.com/) for the web framework.
