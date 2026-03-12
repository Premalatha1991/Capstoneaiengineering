from dotenv import load_dotenv


from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama



from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Load document
loader = TextLoader("data/company_policy.txt")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Create LLM
llm = Ollama(model="llama3")

# Create RAG chain
llm = Ollama(model="llama3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

while True:
    question = input("Question: ")

    if question == "exit":
        break

    result = qa_chain.invoke({"query": question})

    print("Answer:", result["result"])