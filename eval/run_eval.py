import os
import json
import pathlib
from typing import Dict, Any, List
from dotenv import load_dotenv

from agent.graph import agent
from eval.test_set import TEST_SET
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Initialize Gemini Evaluator Judge (temperature=0.0 for deterministic evaluation)
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0
)

# --- LLM-as-a-Judge Prompts ---

RECALL_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Assess whether the retrieved context contains the necessary information to answer the question, based on the Ground Truth statement.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context:
{context}

Respond with ONLY a JSON object containing:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
""")

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Assess if the generated answer is strictly grounded in the provided context and contains no hallucinations or unverified facts.

Context:
{context}

Generated Answer:
{answer}

Respond with ONLY a JSON object containing:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
""")

CORRECTNESS_PROMPT = ChatPromptTemplate.from_template("""
You are an evaluation judge. Compare the generated answer against the ground truth statement for factual and technical accuracy.

Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

Respond with ONLY a JSON object containing:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}}
""")


def parse_llm_json(response_text: str) -> Dict[str, Any]:
    """Helper to parse JSON output safely from LLM responses."""
    cleaned = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"score": 0.0, "reasoning": "Failed to parse evaluation response."}


def evaluate_sample(question: str, ground_truth: str, answer: str, context: str) -> Dict[str, Any]:
    """Evaluates a single sample using Gemini as the judge."""
    
    # 1. Context Recall
    recall_res = (RECALL_PROMPT | evaluator_llm).invoke({
        "question": question,
        "ground_truth": ground_truth,
        "context": context
    })
    recall_eval = parse_llm_json(recall_res.content)

    # 2. Faithfulness
    faith_res = (FAITHFULNESS_PROMPT | evaluator_llm).invoke({
        "context": context,
        "answer": answer
    })
    faith_eval = parse_llm_json(faith_res.content)

    # 3. Answer Correctness
    corr_res = (CORRECTNESS_PROMPT | evaluator_llm).invoke({
        "question": question,
        "ground_truth": ground_truth,
        "answer": answer
    })
    corr_eval = parse_llm_json(corr_res.content)

    return {
        "context_recall": recall_eval.get("score", 0.0),
        "faithfulness": faith_eval.get("score", 0.0),
        "answer_correctness": corr_eval.get("score", 0.0),
        "details": {
            "recall_reasoning": recall_eval.get("reasoning"),
            "faithfulness_reasoning": faith_eval.get("reasoning"),
            "correctness_reasoning": corr_eval.get("reasoning")
        }
    }


def run_evaluation():
    print(f"Starting evaluation on {len(TEST_SET)} test queries using direct Gemini Judge...\n")
    
    results_data = []
    pass_at_1_count = 0
    
    total_recall = 0.0
    total_faithfulness = 0.0
    total_correctness = 0.0

    for idx, test_item in enumerate(TEST_SET, start=1):
        query = test_item["question"]
        ground_truth = test_item["ground_truth"]
        
        print(f"[{idx}/{len(TEST_SET)}] Evaluating Query: '{query}'")
        
        initial_state = {
            "query": query,
            "chunks": [],
            "answer": "",
            "execution_result": "",
            "execution_success": True,
            "retry_count": 0
        }
        
        try:
            output_state = agent.invoke(initial_state)
            answer = output_state.get("answer", "")
            chunks = output_state.get("chunks", [])
            execution_success = output_state.get("execution_success", True)
            retry_count = output_state.get("retry_count", 0)
        except Exception as e:
            print(f"  -> Agent invocation failed: {e}")
            answer = f"Error: {str(e)}"
            chunks = []
            execution_success = False
            retry_count = 2

        is_pass_1 = execution_success and (retry_count == 0)
        if is_pass_1:
            pass_at_1_count += 1

        # Format context for evaluation
        formatted_context = "\n---\n".join([
            f"Function: {c.get('name', 'N/A')}\nSource:\n{c.get('source', '')}" for c in chunks
        ]) if chunks else "No context retrieved."

        # Compute LLM Judge Scores
        eval_scores = evaluate_sample(query, ground_truth, answer, formatted_context)
        
        total_recall += eval_scores["context_recall"]
        total_faithfulness += eval_scores["faithfulness"]
        total_correctness += eval_scores["answer_correctness"]

        results_data.append({
            "question": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "execution_success": execution_success,
            "retry_count": retry_count,
            "pass_at_1": is_pass_1,
            "scores": eval_scores
        })
        print(f"  -> Pass@1: {is_pass_1} | Recall: {eval_scores['context_recall']:.2f} | Faithfulness: {eval_scores['faithfulness']:.2f} | Correctness: {eval_scores['answer_correctness']:.2f}")

    num_samples = len(TEST_SET)
    pass_at_1_score = pass_at_1_count / num_samples
    avg_recall = total_recall / num_samples
    avg_faithfulness = total_faithfulness / num_samples
    avg_correctness = total_correctness / num_samples

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Pass@1 Rate          : {pass_at_1_score:.2%} ({pass_at_1_count}/{num_samples})")
    print(f"Avg Context Recall   : {avg_recall:.4f}")
    print(f"Avg Faithfulness     : {avg_faithfulness:.4f}")
    print(f"Avg Answer Correctness: {avg_correctness:.4f}")
    print("=" * 60)

    # Save to eval/results.json
    eval_dir = pathlib.Path("eval")
    eval_dir.mkdir(exist_ok=True)
    results_file = eval_dir / "results.json"
    
    summary_output = {
        "pass_at_1": pass_at_1_score,
        "metrics_summary": {
            "context_recall": avg_recall,
            "faithfulness": avg_faithfulness,
            "answer_correctness": avg_correctness
        },
        "detailed_results": results_data
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=4)

    print(f"\nEvaluation results saved successfully to {results_file}")

if __name__ == "__main__":
    run_evaluation()