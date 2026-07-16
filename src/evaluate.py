import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from chat import answer_question  # reuses your Day 4 RAG pipeline

load_dotenv()

openai_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


def judge_response(question, answer, context_chunks):
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    judge_prompt = f"""You are an impartial evaluator of a RAG system's answer.

Question: {question}

Retrieved Context:
{context_text}

Generated Answer: {answer}

Evaluate the answer on these criteria and respond ONLY in JSON format, no other text:
{{
  "grounded": true/false,
  "relevant": true/false,
  "appropriately_refused": true/false or null,
  "issues": "brief description of any problems, or 'none'"
}}"""

    response = openai_client.chat.completions.create(
        model=chat_deployment,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Could not parse judge response", "raw": raw}


if __name__ == "__main__":
    with open(os.path.join("data", "eval_questions.json"), "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []

    for item in eval_set:
        question = item["question"]
        print(f"Evaluating: {question}")

        answer, chunks = answer_question(question)
        verdict = judge_response(question, answer, chunks)

        results.append({
            "question": question,
            "answer": answer,
            "verdict": verdict
        })

        print(f"  → Grounded: {verdict.get('grounded')}, Relevant: {verdict.get('relevant')}, Issues: {verdict.get('issues')}\n")

    output_path = os.path.join("data", "eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    grounded_count = sum(1 for r in results if r["verdict"].get("grounded") is True)
    print(f"\n=== Summary ===")
    print(f"Grounded: {grounded_count}/{len(results)}")
    print(f"Full report saved to {output_path}")