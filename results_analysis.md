# Evaluation Analysis & Findings

## 1. Performance Overview
We evaluated the RAG system using three chunking strategies on a dataset of 25 questions related to Dr. Ambedkar's works. The goal was to identify the optimal configuration for accuracy and answer quality.

### Metric Comparison Table

| Metric | Small (250 chars) | Medium (550 chars) | Large (900 chars) |
| :--- | :--- | :--- | :--- |
| **Hit Rate** (Retrieval) | 1.00 | 1.00 | 1.00 |
| **MRR** (Rank Accuracy) | 0.90 | 0.90 | 0.90 |
| **ROUGE-L** (Factuality) | 0.2552 | **0.2948** | 0.2716 |
| **BLEU Score** (Precision) | 0.1845 | **0.2145** | 0.1981 |
| **Cosine Sim** (Semantic) | 0.5835 | 0.5755 | **0.5899** |

## 2. Analysis of Strategies

### Small Chunks (250 chars)
* **Performance:** Lowest scores in answer quality (ROUGE 0.25 / BLEU 0.18).
* **Observation:** While retrieval was perfect (Hit Rate 1.0), the generated answers often lacked depth. The small window size likely fragmented sentences, causing the LLM to miss the broader context needed to form complete answers.

### Medium Chunks (550 chars) - **WINNER**
* **Performance:** **Highest scores** in ROUGE (0.2948) and BLEU (0.2145).
* **Observation:** This size proved to be the "Sweet Spot." It provided enough context for the LLM to understand the *relationship* between ideas (essential for questions like Q2 and Q3) without overwhelming the context window with irrelevant text.

### Large Chunks (900 chars)
* **Performance:** Strong semantic similarity (Cosine 0.59) but lower word-for-word accuracy than Medium.
* **Observation:** Large chunks captured the general "vibe" or meaning well (high Cosine Similarity), but the extra noise in the retrieved text likely confused the specific phrasing of the answer, leading to a lower BLEU score.

## 3. Common Failure Modes

Based on the detailed results, the system struggled most with **Conceptual** and **Comparative** questions:

1.  **Complex Synthesis Failure (e.g., Q20):**
    * *Question:* "Relationship between education and liberation..."
    * *Score:* ROUGE dropped to ~0.07-0.09 across all strategies.
    * *Reason:* The answer required synthesizing philosophy from multiple places, which simple RAG (retrieving specific text chunks) struggles to do effectively.

2.  **Comparative Logic (e.g., Q9):**
    * *Question:* "Relate political power to social change in Docs 3 and 6."
    * *Score:* Low ROUGE (~0.10).
    * *Reason:* The system retrieved the documents correctly but failed to explicitly structure the comparison in the final answer, leading to a mismatch with the structured ground truth.

## 4. Recommendations

Based on this evaluation, the **Medium Strategy (550 chars)** is the optimal configuration for this corpus.

### Specific Improvements to Boost Performance:
1.  **Implement Re-ranking:** Although retrieval Hit Rate is high, MRR is 0.90 (not perfect). A re-ranker (like BGE-Reranker) would ensure the *most* relevant chunk is always #1, helping with complex questions.
2.  **Chain-of-Thought Prompting:** For comparative questions (Q7, Q9, Q18), modify the system prompt to ask the LLM to "Step 1: Analyze Doc A, Step 2: Analyze Doc B, Step 3: Compare."
3.  **Adjust Overlap:** For the Medium strategy, increasing chunk overlap from 50 to 100 characters might prevent vital sentences from being cut in half, potentially improving the BLEU score further.