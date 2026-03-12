# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

project = "rag_project"

files = {
"app.py": """
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

DATA_PATH = "data/company_policy.txt"

def load_docs():
    loader = TextLoader(DATA_PATH)
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def main():
    docs = load_docs()
    docs = split_docs(docs)

    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    while True:
        q = input("Ask question: ")
        if q == "exit":
            break
        print(qa.run(q))

if __name__ == "__main__":
    main()
""",

"requirements.txt": """
langchain
langchain-community
langchain-openai
faiss-cpu
tiktoken
python-dotenv
""",

"data/company_policy.txt": """
Company Policy

Leave Policy:
Employees get 20 paid leave days per year.

Work Hours:
9 AM to 6 PM Monday to Friday.

Remote Work:
Allowed 2 days per week with manager approval.
"""
}

os.makedirs(project + "/data", exist_ok=True)

for path, content in files.items():
    full = os.path.join(project, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)

print("✅ RAG project created successfully!")
