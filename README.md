# Project README

This repository contains several Flask-based web services that leverage different natural language processing (NLP) models and tools for various tasks. Below is an overview of each project included in this repository along with their functionalities and usage instructions.

## 1. Llama Index - Query Engine

This project implements a Flask-based web service that utilizes the LLama Index for efficient querying of documents. The LLama Index integrates LLama2, Hugging Face Embeddings, and other tools for text processing and searching.

### Usage Instructions

1. Install the required dependencies listed in the `requirements.txt` file.
2. Run the Flask app using `python app.py`.
3. Send POST requests to the `/query` endpoint with a JSON payload containing the 'query' key to retrieve relevant information from the document index.

## 2. OpenAI Chat - Conversational Model

The OpenAI Chat project integrates the OpenAI GPT-3.5 model for conversational AI capabilities. It provides an API endpoint for generating responses based on user queries using the OpenAI model.

### Usage Instructions

1. Ensure you have an OpenAI API key and set it in the environment variable `OPENAI_API_KEY`.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Run the Flask app using `python app.py`.
4. Send POST requests to the `/query` endpoint with a JSON payload containing the 'query' key to generate responses using the OpenAI model.

## 3. Sentence Transformer - Embedding Model

The Sentence Transformer project implements a web service that utilizes the Hugging Face Sentence Transformers for generating sentence embeddings. It provides an API endpoint for computing embeddings for input sentences.

### Usage Instructions

1. Install the required dependencies listed in the `requirements.txt` file.
2. Run the Flask app using `python app.py`.
3. Send POST requests to the `/embed` endpoint with a JSON payload containing the 'sentence' key to compute embeddings for the input sentences.

## 4. Amazon Comprehend - NLP Service

The Amazon Comprehend project integrates the Amazon Comprehend service for natural language processing tasks. It provides endpoints for querying knowledge bases and extracting information using the Amazon Comprehend API.

### Usage Instructions

1. Install the required dependencies listed in the `requirements.txt` file.
2. Set up AWS credentials and ensure the `boto3` library can access the Amazon Comprehend service.
3. Run the Flask app using `python app.py`.
4. Send POST requests to the `/bedrock` and `/sample` endpoints with JSON payloads containing the 'query' key for querying knowledge bases and obtaining responses from Amazon Comprehend.

---

Feel free to expand on these instructions with details on deployment, additional configurations, or any specific usage guidelines for each project.
