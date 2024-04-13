from flask import Flask,Blueprint, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import SimpleDirectoryReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
app = Flask(__name__)

# Load PDF and prepare the necessary components
loader = PyPDFLoader("./sample_data/pdf_samples")
docs = loader.load()

# docs = SimpleDirectoryReader("./data").load_data()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Create vector store and retriever
db = FAISS.from_documents(documents[:30], OpenAIEmbeddings())
retriever = db.as_retriever()

# Load LLM model and create ChatPromptTemplate
llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
from flask import Flask,Blueprint, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core import SimpleDirectoryReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
app = Flask(__name__)

# Load PDF and prepare the necessary components
loader = PyPDFLoader("./sample_data/pdf_samples")
docs = loader.load()

# docs = SimpleDirectoryReader("./data").load_data()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Create vector store and retriever
db = FAISS.from_documents(documents[:30], OpenAIEmbeddings())
retriever = db.as_retriever()

# Load LLM model and create ChatPromptTemplate
llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

# Create document chain and retrieval chain
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Created the Blueprint
routes = Blueprint('llama_openai_model', __name__)

@routes.route('/answer_question', methods=['POST'])
def answer_question():
    query = request.json['query']
    response = retrieval_chain.invoke({"input": query})
    return jsonify({'answer': response['answer']})


# if __name__ == '__main__':
#     app.run(port= 5000)
# Created the Blueprint
routes = Blueprint('llama_openai_model', __name__)

@routes.route('/answer_question', methods=['POST'])
def answer_question():
    query = request.json['query']
    response = retrieval_chain.invoke({"input": query})
    return jsonify({'answer': response['answer']})


# if __name__ == '__main__':
#     app.run(port= 5000)