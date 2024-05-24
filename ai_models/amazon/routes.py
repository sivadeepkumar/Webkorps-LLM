import boto3
from flask import render_template, request,jsonify
from flask import Blueprint
from .helper import BedrockKBAgent
import os 

# aws_ai = Flask(__name__)
aws_ai = Blueprint('amazon_model', __name__)
kb = BedrockKBAgent()
client = boto3.client('comprehend', region_name='us-east-1')
from flask_restx import Resource
from ai_models.open_ai.routes import api,query_model,query_model_source,query_model_form
from langchain_community.vectorstores.faiss import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3

# Initialize AWS services and models
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



region_name = 'us-east-1' 

class bedrock_tech(Resource):
    @api.expect(query_model)
    def post(self):
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
        try:
            data = request.get_json()
            query = data.get('query')
            if not query:
                raise ValueError("Please enter a valid input.")
            
            kb_id = os.getenv("kb_id")  #  or "QRJWFQFERS"
            response = kb.retrieve_from_kb(kb_id, query)
            response =  response['retrievalResults'] # [0]['content']
            
            success_message = {
                                    "status": "Success",
                                    "Response": response
                            }
            return jsonify(success_message)
                    
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)
            }
            return jsonify(failure_response)







@aws_ai.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200


# manual_ingestion
def manual_vector_store(docs):
    """
    Manually creates a vector store from the provided documents using FAISS.

    Args:
    - docs (list): List of documents for which vectors need to be created.

    Returns:
    - None
    """

    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("sample_data/manual_index")
    


def manual_ingestion(source):
    """
    Performs manual ingestion of a single source for text processing.

    Args:
    - source (str): The source text to be processed.

    Returns:
    - list: List of processed documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.create_documents([source])
    return docs

# Data ingestion function
def data_ingestion():
    """
    Loads documents from PDF samples directory and performs text splitting.

    Returns:
    - list: List of split documents.
    """
    
    # pdf_path = '../../sample_data/pdf_samples'
    loader = PyPDFDirectoryLoader("sample_data/pdf_samples")
    # import pdb ; pdb.set_trace()
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector embedding and vector store function
def get_vector_store(docs):
    """
    Creates a vector store from the provided documents using FAISS.

    Args:
    - docs (list): List of documents for which vectors need to be created.

    Returns:
    - None
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("sample_data/faiss_index")

# LLM models functions
def get_mistral_llm():
    """
    Retrieves the Mistral LLM model from Amazon Bedrock.

    Returns:
    - Bedrock: Mistral LLM model instance.
    """
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock, model_kwargs={'max_tokens': 1024})
    return llm

def get_llama3_llm():
    """
    Retrieves the Llama2 LLM model from Amazon Bedrock.

    Returns:
    - Bedrock: Llama2 LLM model instance.
    """
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
    """
    Retrieves a response from the LLM model using the specified query and vector store.

    Args:
    - llm (Bedrock): LLM model instance.
    - vectorstore_faiss (FAISS): Vector store created from documents.
    - query (str): User query to be processed.

    Returns:
    - str: Processed response from the LLM model.
    """
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                     retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                                     return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    answer = qa(query)
    answer = answer['result']

    if answer.startswith("Answer: "):
        answer = answer[len("Answer: "):]
    return answer

# Route for updating vector base using GET request
class update_vector_base(Resource):
    def get(self):
        """
        Updates the vector base using a GET request.

        Returns:
        - str: Success message if the vector base is updated successfully.
        """
        docs = data_ingestion()
        get_vector_store(docs)
        return 'Vectors Updated Successfully'



class mistral_response(Resource):
    @api.expect(query_model)
    def post(self):
        """
        Handles POST requests for Mistral model responses.

        Returns:
        - JSON: Response message containing status and processed data.
        """
        try:
            user_question = request.json['query']

            if not user_question:
                raise ValueError("Please enter a valid input.")

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
            "Response": str(e)

            }
            return jsonify(failure_response)



# Route for Llama2 model response using POST request
class llama3_response(Resource):
    @api.expect(query_model)
    def post(self):
        """
        Handles POST requests for Llama2 model responses.

        Returns:
        - JSON: Response message containing status and processed data.
        """
        try:
            user_question = request.json['query']

            if not user_question:
                raise ValueError("Please enter a valid input.")

            faiss_index = FAISS.load_local("sample_data/faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)
            success_message = {
                                    "status": "Success",
                                    "Response": response
                            }
            return jsonify(success_message)
                    
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)

            }
            return jsonify(failure_response)



class llama_source(Resource):
    @api.expect(query_model_source)
    def post(self):
        """
        Endpoint for querying data using Llama3 LLM.

        Retrieves a user query and source data, performs a similarity search, and runs a QA chain to generate a response.
        """
        try:
            data = request.get_json()
            query = data['query']
            source = data['source']

            if not query:
                raise ValueError("Please enter a valid input.")
            

            # Load embeddings model (replace 'Your_Embeddings_Model' with the actual name of your embeddings model)
            embeddings = bedrock_embeddings

            # Load LLM from Amazon Bedrock (replace 'get_llm' with the actual function for loading Amazon Bedrock LLM)
            llm = get_llama3_llm()

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
            "Response": str(e)
            }
            return jsonify(failure_response)




class mistral_source(Resource):
    @api.expect(query_model_source)
    def post(self):
        """
        Endpoint for querying data using Mistral LLM.

        Retrieves a user query and source data, performs a similarity search, and runs a QA chain to generate a response.
        """ 
        try:
            data = request.get_json()
            query = data['query']
            source = data['source']

            # Check if user_question is empty
            if not query:
                raise ValueError("Please enter a valid input.")

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
            "Response": str(e)
            }
            return jsonify(failure_response)





def get_first_embedding(source_data):
    """
    Processes source data to obtain the first embedding for a question.

    This function prepares the source data, creates a FAISS index, and retrieves the first embedding for a user question.
    """
    
    user_question = "Which form we need to create just one word answer like <Create the ________ form> in this format,I need to get the response"
    docs = manual_ingestion(source_data)
    manual_vector_store(docs)
    faiss_index = FAISS.load_local("sample_data/manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    llm = get_mistral_llm()
    return get_response_llm(llm, faiss_index, user_question)

        

class mistral_form(Resource):
    @api.expect(query_model_form)
    def post(self):
        """
        Endpoint for generating form-related information using Llama3 LLM.

        Retrieves a user question and source data, generates the first embedding, and uses it to request relevant information.
        """
        try:
            user_question = request.json['query']
            source_data = request.json['source']
            type = request.json['type']

            # Check if user_question is empty
            if not user_question:
                raise ValueError("Please enter a valid input.")
            

            query = get_first_embedding(user_question)
            print(query)
            
            docs = manual_ingestion(source_data)
            manual_vector_store(docs)


            if type == "create":
                user_question = f"provide me the relevant columns name in array only for {query} would be."
            else:
                target_word = query.replace("form", "")
                user_question = f"provide me the relevant columns name in array only for updating {target_word} would be."

            
            
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
            "Response": str(e)
            }
            return jsonify(failure_response)




class llama_form(Resource):
    @api.expect(query_model_form)
    def post(self):
        """
        Endpoint for generating form-related information using Llama3 LLM.

        Retrieves a user question and source data, generates the first embedding, and uses it to request relevant information.
        """
        try:
            user_question = request.json['query']
            source_data = request.json['source']
            type = request.json['type']

            # Check if user_question is empty
            if not user_question:
                raise ValueError("Please enter a valid input.")

            query = get_first_embedding(user_question)
            print(query)
            
            docs = manual_ingestion(source_data)
            manual_vector_store(docs)


            if type == "create":
                user_question = f"provide me the relevant columns name in array only for {query} would be."
            else:
                target_word = query.replace("form", "")
                user_question = f"provide me the relevant columns name in array only for updating {target_word} would be."

            
            
            # FROM STORED DATA IT WILL RETRIEVE 
            faiss_index = FAISS.load_local("sample_data/manual_index", bedrock_embeddings, allow_dangerous_deserialization=True)

            # Assuming these functions are defined elsewhere
            llm = get_llama3_llm()
            response = get_response_llm(llm, faiss_index, user_question)


            success_message = {
                    "status": "Success",
                    "Response": response
                }
            return jsonify(success_message)
        
        except Exception as e:
            failure_response = {
            "status": "Failure",
            "Response": str(e)
            }
            return jsonify(failure_response)





api.add_resource(update_vector_base,'/amazon_model/update_vector_base')
api.add_resource(mistral_response,'/amazon_model/mistral_response')
api.add_resource(mistral_source,'/amazon_model/mistral/source')
api.add_resource(mistral_form,'/amazon_model/mistral/form')
api.add_resource(llama3_response,'/amazon_model/llama_response')
api.add_resource(llama_source,'/amazon_model/llama/source')
api.add_resource(llama_form,'/amazon_model/llama/form')
api.add_resource(bedrock_tech,'/amazon_model/bedrock')
