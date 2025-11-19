from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

def main():
    print("--- Initialize System ---")
    
    loader = TextLoader("speech.txt")
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating embeddings and vector store (this might take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(chunks, embeddings)

    llm = Ollama(model="mistral")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    print("\n--- Ready! Ask a question about the speech (or type 'exit' to quit) ---")
    
    while True:
        user_query = input("\nYour Question: ")
        
        if user_query.lower() == 'exit':
            break
            

        response = qa_chain.run(user_query)

        print(f"Answer: {response}")

if __name__ == "__main__":
    main()