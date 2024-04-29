import boto3
from flask import Flask, render_template, request,jsonify
from flask import Blueprint
from .helper import BedrockKBAgent
import os 

# aws_ai = Flask(__name__)
aws_ai = Blueprint('amazon_model', __name__)
kb = BedrockKBAgent()
client = boto3.client('comprehend', region_name='us-east-1')


@aws_ai.route('/bedrock', methods=['POST'])
def bedrock_tech():
    """
        Handles queries related to a specific knowledge base using BedrockKBAgent.

        This endpoint takes a POST request with a JSON payload containing the 'query' key representing the user's query. It uses BedrockKBAgent to retrieve information from a specific knowledge base identified by the 'kb_id' variable and returns the retrieval results in JSON format.

        Parameters:
            None (Request payload contains the 'query' key with the user's query string).

        Returns:
            JSON: A JSON response containing the retrieval results from the specified knowledge base.

        Raises:
            None (Any errors in retrieving data from the knowledge base are handled internally within BedrockKBAgent).
    """

    data = request.get_json()
    query = data.get('query')
    kb_id = os.getenv("kb_id")  #  or "QRJWFQFERS"
    response = kb.retrieve_from_kb(kb_id, query)
    return response['retrievalResults'] # [0]['content']



from flask import Flask, request, jsonify, g
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

# Initialize AWS services and models
region_name = 'us-east-1' 
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client = bedrock)

from dotenv import load_dotenv
from flask_cors import CORS 
import os
import logging
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





@aws_ai.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200










# manual_ingestion
def manual_vector_store(docs):
    # Assuming bedrock_embeddings is defined somewhere
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("sample_data/manual_index")
    


def manual_ingestion(source):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([source])
    return docs

# Data ingestion function
def data_ingestion():
    
    # pdf_path = '../../sample_data/pdf_samples'
    loader = PyPDFDirectoryLoader("sample_data/pdf_samples")
    # import pdb ; pdb.set_trace()s
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    # pdb.set_trace()
    return docs

# Vector embedding and vector store function
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("sample_data/faiss_index")

# LLM models functions
def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 1024})
    return llm

def get_llama2_llm():
    return Bedrock(
        model_id="meta.llama3-70b-instruct-v1:0",
        client=bedrock,
        model_kwargs={'max_gen_len': 512},
    )

# Prompt template for LLM responses
prompt_template = """
{context}
Question: {question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to get response from LLM model
def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                                     return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    answer = qa(query)
    answer = answer['result']

    if answer.startswith("Answer: "):
        answer = answer[len("Answer: "):]
    return answer

# Route for updating vector base using GET request
@aws_ai.route('/update_vector_base', methods=['GET'])
def update_vector_base():
    docs = data_ingestion()
    get_vector_store(docs)
    return 'Vectors Updated Successfully'

# Route for Mistral model response using POST request
@aws_ai.route('/mistral_response', methods=['POST'])
def mistral_response():
    try:
        user_question = request.json['query']
        faiss_index = FAISS.load_local("sample_data/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        llm = get_mistral_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        success_message = {
                                "status": "Success",
                                "Response": response
                        }
        return jsonify(success_message)
                
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)

# Route for Llama2 model response using POST request
@aws_ai.route('/llama_response', methods=['POST'])
def llama2_response():
    try:
        user_question = request.json['query']
        faiss_index = FAISS.load_local("sample_data/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        llm = get_llama2_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        success_message = {
                                "status": "Success",
                                "Response": response
                        }
        return jsonify(success_message)
                
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)



@aws_ai.route('/llama/source', methods=['POST'])
def llama_source():
    try:
        data = request.get_json()
        query = data['query']
        source = data['source']

        # Load embeddings model (replace 'Your_Embeddings_Model' with the actual name of your embeddings model)
        embeddings = bedrock_embeddings

        # Load LLM from Amazon Bedrock (replace 'get_llm' with the actual function for loading Amazon Bedrock LLM)
        llm = get_llama2_llm()

        # Create FAISS index from source data
        document_search = FAISS.from_texts([source], embeddings)

        # Assuming load_qa_chain is a function to load your QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Perform similarity search and run the QA chain
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        success_message = {
                            "status": "Success",
                            "Response": result
                    }
        return jsonify(success_message)
            
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)

@aws_ai.route('/mistral/source', methods=['POST'])
def mistral_source():
    try:
        data = request.get_json()
        query = data['query']
        source = data['source']

        # Load embeddings model (replace 'Your_Embeddings_Model' with the actual name of your embeddings model)
        embeddings = bedrock_embeddings

        # Load LLM from Amazon Bedrock (replace 'get_llm' with the actual function for loading Amazon Bedrock LLM)
        llm = get_mistral_llm()

        # Create FAISS index from source data
        document_search = FAISS.from_texts([source], embeddings)

        # Assuming load_qa_chain is a function to load your QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Perform similarity search and run the QA chain
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        success_message = {
                        "status": "Success",
                        "Response": result
                }
        return jsonify(success_message)
        
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)

def get_first_embedding(source_data):
    try:
        user_question = "Which form we need to create just one word answer like <Create the ________ form> in this format,I need to get the response"
        docs = manual_ingestion(source_data)
        manual_vector_store(docs)
        faiss_index = FAISS.load_local("sample_data/manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)
        llm = get_mistral_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        
        success_message = {
                    "status": "Success",
                    "Response": response
                }
        return jsonify(success_message)
        
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)

@aws_ai.route('/mistral/form', methods=['POST'])
def mistral_form():
    try:
        user_question = request.json['query']
        source_data = request.json['source']

        query = get_first_embedding(user_question)

        print(query)
        user_question = f"provide me the relevant columns name in array only for {query} would be."
        docs = manual_ingestion(source_data)
        manual_vector_store(docs)

        
        # FROM STORED DATA IT WILL RETRIEVE 
        faiss_index = FAISS.load_local("sample_data/manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

        # Assuming these functions are defined elsewhere
        llm = get_mistral_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        success_message = {
                "status": "Success",
                "Response": response
            }
        return jsonify(success_message)
    
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)



@aws_ai.route('/llama/form', methods=['POST'])
def llama_form():
    try:
        user_question = request.json['query']
        source_data = request.json['source']

        query = get_first_embedding(user_question)
        print(query)
        user_question = f"provide me the relevant columns name in array only for {query} would be."
        docs = manual_ingestion(source_data)
        manual_vector_store(docs)

        
        # FROM STORED DATA IT WILL RETRIEVE 
        faiss_index = FAISS.load_local("sample_data/manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

        # Assuming these functions are defined elsewhere
        llm = get_llama2_llm()
        response = get_response_llm(llm, faiss_index, user_question)
        success_message = {
                "status": "Success",
                "Response": response
            }
        return jsonify(success_message)
    
    except Exception as e:
        failure_response = {
        "status": "Failure",
        "Response": e
        }
        return jsonify(failure_response)




