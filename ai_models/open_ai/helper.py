# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import logging

logging.basicConfig(filename='./logs/model_service.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def process_query(file_name, query):
    """
    Processes a query using a specified file and AI models.

    This function takes two parameters: 'file_name', which is the name of the file containing text samples, and 'query', which is the query to be processed. The function performs the following steps:
    1. Initializes OpenAI embeddings for natural language processing.
    2. Loads text samples from the specified file.
    3. Performs similarity search using FAISS (Facebook AI Similarity Search) based on the embeddings and text samples.
    4. Loads a QA (Question Answering) chain model using OpenAI.
    5. Searches for relevant documents based on the query using the similarity search results.
    6. Runs the QA chain model to generate a response to the query based on the input documents.

    Parameters:
        file_name (str): The name of the file containing text samples to be searched.
        query (str): The query to be processed and answered.
(llm_model) dell@user-Inspiron-15-3511:~/Documents/new_webkorps_llm/new_webkorps_llm$ code .
(llm_model) dell@user-Inspiron-15-3511:~/Documents/new_webkorps_llm/new_webkorps_llm$ python3 app.py
Traceback (most recent call last):
  File "/home/dell/Documents/new_webkorps_llm/new_webkorps_llm/app.py", line 31, in <module>
    kb = BedrockKBAgent(setup=True)
  File "/home/dell/Documents/new_webkorps_llm/new_webkorps_llm/ai_models/aws/helper.py", line 21, in __init__
    self.setup()
  File "/home/dell/Documents/new_webkorps_llm/new_webkorps_llm/ai_models/aws/helper.py", line 49, in setup
    response_kb = kb.create_kb(kb_name="bedrock-ra-01",
NameError: name 'kb' is not defined

    Returns:
        dict: A dictionary containing the processed result generated in response to the query.
        
    Raises:
        Exception: If any error occurs during the processing of the query, an exception is raised and logged.
    """
    try:
        # Initialize embeddings and load documents
        embeddings = OpenAIEmbeddings()
        root_path = './sample_data/text_samples/'
        full_path = root_path + file_name
        with open(full_path, 'r') as f:
            texts = f.read()
            
        # Perform FAISS search
        searched_documents = FAISS.from_texts([texts], embeddings)
        # Load QA chain model
        model_chain = load_qa_chain(OpenAI(), chain_type="stuff")
        # Perform similarity search
        docs = searched_documents.similarity_search(query)
        # Run the QA chain model
        result = model_chain.run(input_documents=docs, question=query)
        
        return result
    except Exception as e:
        logger.exception("Error in processing query."+str(e)+"file_name: "+file_name+"query: "+query)
