import json
import os
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# --- FIX: Download ALL necessary NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
# ---------------------------------------------

# --- CONFIGURATION ---
CHUNK_STRATEGIES = {
    "Small": 250,   # 200-300 range
    "Medium": 550,  # 500-600 range
    "Large": 900    # 800-1000 range
}
MODEL_NAME = "mistral"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize Embeddings once (used for both retrieval and evaluation)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def calculate_cosine_similarity(text1, text2):
    """Calculates semantic similarity between two texts using embeddings."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vec1 = embedding_model.embed_query(text1)
        vec2 = embedding_model.embed_query(text2)
        return cosine_similarity([vec1], [vec2])[0][0]
    except:
        return 0.0

def calculate_retrieval_metrics(retrieved_docs, expected_sources):
    """
    Calculates Hit Rate, MRR, and Precision based on file names.
    """
    retrieved_sources = [doc.metadata['source'].split('/')[-1] if '/' in doc.metadata['source'] else doc.metadata['source'].split('\\')[-1] for doc in retrieved_docs]
    expected_sources = [s.split('/')[-1] for s in expected_sources] # Normalize paths
    
    # 1. Hit Rate (Is at least one correct doc retrieved?)
    hit = any(source in expected_sources for source in retrieved_sources)
    
    # 2. MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, source in enumerate(retrieved_sources):
        if source in expected_sources:
            mrr = 1 / (i + 1)
            break
            
    # 3. Precision (How many of the retrieved docs are relevant?)
    relevant_retrieved = sum(1 for s in retrieved_sources if s in expected_sources)
    precision = relevant_retrieved / len(retrieved_sources) if retrieved_sources else 0
    
    return hit, mrr, precision

def calculate_quality_metrics(generated_answer, ground_truth):
    """Calculates ROUGE, BLEU and Cosine Similarity."""
    
    # 1. ROUGE Score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(ground_truth, generated_answer)['rougeL'].fmeasure

    # 2. BLEU Score
    ref_tokens = nltk.word_tokenize(ground_truth.lower())
    cand_tokens = nltk.word_tokenize(generated_answer.lower())
    if not cand_tokens:
        bleu_score = 0
    else:
        try:
            bleu_score = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0))
        except:
            bleu_score = 0

    # 3. Cosine Similarity
    sem_score = calculate_cosine_similarity(generated_answer, ground_truth)

    return {
        "rouge_l": round(rouge_score, 4),
        "bleu_score": round(bleu_score, 4),
        "cosine_similarity": round(sem_score, 4)
    }

def evaluate_strategy(strategy_name, chunk_size, documents, test_data):
    print(f"\n=== Evaluating Strategy: {strategy_name} (Chunk Size: {chunk_size}) ===")
    
    # 1. Setup Vector Store for this strategy
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 docs
    
    # 2. Setup Chain (return_source_documents=True is CRITICAL for metrics)
    llm = Ollama(model=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )

    results = []
    metrics_summary = {
        "total_hits": 0,
        "total_mrr": 0,
        "total_precision": 0,
        "total_rouge": 0,
        "total_bleu": 0,
        "total_cosine": 0
    }

    # 3. Run Test Questions
    for item in test_data:
        # Skip unanswerable questions for retrieval accuracy context
        if not item.get('answerable', True):
            continue

        question = item['question']
        ground_truth = item['ground_truth']
        expected_sources = item.get('source_documents', [])
        
        try:
            response = qa_chain.invoke(question)
            generated_answer = response['result']
            retrieved_docs = response['source_documents']
            
            # --- Calculate Metrics ---
            # Retrieval
            hit, mrr, precision = calculate_retrieval_metrics(retrieved_docs, expected_sources)
            
            # Quality
            quality_metrics = calculate_quality_metrics(generated_answer, ground_truth)
            
            # Update Totals
            metrics_summary['total_hits'] += 1 if hit else 0
            metrics_summary['total_mrr'] += mrr
            metrics_summary['total_precision'] += precision
            metrics_summary['total_rouge'] += quality_metrics['rouge_l']
            metrics_summary['total_bleu'] += quality_metrics['bleu_score']
            metrics_summary['total_cosine'] += quality_metrics['cosine_similarity']
            
            # Log individual result
            results.append({
                "id": item['id'],
                "question": question,
                "retrieval_metrics": {"hit": hit, "mrr": mrr, "precision": precision},
                "quality_metrics": quality_metrics
            })
            print(f"Q{item['id']}: Hit={hit} | ROUGE={quality_metrics['rouge_l']:.2f} | Cosine={quality_metrics['cosine_similarity']:.2f}")

        except Exception as e:
            print(f"Error on Q{item['id']}: {e}")

    # 4. Cleanup
    vector_store.delete_collection()
    
    # 5. Calculate Averages
    count = len(results) if results else 1
    averages = {
        "avg_hit_rate": metrics_summary['total_hits'] / count,
        "avg_mrr": metrics_summary['total_mrr'] / count,
        "avg_precision": metrics_summary['total_precision'] / count,
        "avg_rouge_l": metrics_summary['total_rouge'] / count,
        "avg_bleu": metrics_summary['total_bleu'] / count,
        "avg_cosine_similarity": metrics_summary['total_cosine'] / count
    }
    
    print(f"Strategy {strategy_name} Finished. Avg Hit Rate: {averages['avg_hit_rate']:.2f}")
    return results, averages

def main():
    # Load Documents
    print("Loading Corpus...")
    loader = DirectoryLoader('./corpus', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Load Test Data
    print("Loading Test Dataset...")
    with open('test_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        test_data = data['test_questions']

    final_output = {}

    # Loop through strategies (Small, Medium, Large)
    for strategy_name, chunk_size in CHUNK_STRATEGIES.items():
        detailed_results, avg_metrics = evaluate_strategy(strategy_name, chunk_size, documents, test_data)
        final_output[strategy_name] = {
            "chunk_size": chunk_size,
            "metrics": avg_metrics,
            "detailed_results": detailed_results
        }

    # Save Final JSON
    with open('test_results.json', 'w') as f:
        json.dump(final_output, f, indent=4)
    
    print("\nAll evaluations complete. Results saved to 'test_results.json'.")

if __name__ == "__main__":
    main()