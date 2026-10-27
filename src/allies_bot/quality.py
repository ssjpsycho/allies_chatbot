"""Data quality report for the knowledge base.

Offline check (no OpenAI/Qdrant calls, only BookStack):
    python3 -m allies_bot.quality --wiki [--show-page "Page name"]

Retrieval check against the live index, using a JSON file of
[{"question": "...", "expected": "Page title"}, ...]:
    python3 -m allies_bot.quality --queries questions.json
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from allies_bot.config import Settings
from allies_bot.ingest import fetch_epub, fetch_wiki
from allies_bot.knowledge import CHUNK_SIZE, DocumentChunk, KnowledgeBase, chunk_document

TINY_CHUNK_CHARS = 150
MIN_BODY_CHARS = 40


def analyze_documents(documents: list[DocumentChunk]) -> dict:
    total_chunks = 0
    sizes: list[int] = []
    issues: list[dict[str, str]] = []
    for document in documents:
        chunks = chunk_document(document.source_label, document.text)
        if not chunks:
            issues.append({"page": document.source_label, "issue": "no indexable text"})
            continue
        total_chunks += len(chunks)
        if not re.search(r"^#{1,6} ", document.text, re.MULTILINE) and len(document.text) > CHUNK_SIZE:
            issues.append({"page": document.source_label, "issue": "long page without headings"})
        for text, heading in chunks:
            sizes.append(len(text))
            body = text.split("\n\n", 1)[-1]
            if len(body) < MIN_BODY_CHARS:
                issues.append({"page": document.source_label, "issue": f"tiny chunk: {body!r}"})
            elif len(text) < TINY_CHUNK_CHARS:
                issues.append({"page": document.source_label, "issue": f"short chunk under '{heading}'"})
            letters = sum(char.isalnum() or char.isspace() for char in body)
            if body and letters / len(body) < 0.6:
                issues.append({"page": document.source_label, "issue": f"garbled text: {body[:60]!r}"})
    return {
        "documents": len(documents),
        "chunks": total_chunks,
        "avg_chunk_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "max_chunk_chars": max(sizes, default=0),
        "issues": issues,
    }


def evaluate_queries(knowledge: KnowledgeBase, cases: list[dict[str, str]]) -> list[dict]:
    results = []
    for case in cases:
        sources = knowledge.search(case["question"])
        labels = [str(source.get("source_label")) for source in sources]
        expected = case["expected"].lower()
        rank = next((i + 1 for i, label in enumerate(labels) if expected in label.lower()), None)
        results.append({"question": case["question"], "expected": case["expected"], "rank": rank, "top": labels[:5]})
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Report on knowledge base data quality.")
    parser.add_argument("--wiki", action="store_true", help="Analyze chunking of the BookStack wiki.")
    parser.add_argument("--epub", action="store_true", help="Analyze chunking of the EPUB.")
    parser.add_argument("--show-page", help="Print the chunks of the page with this exact name.")
    parser.add_argument("--queries", type=Path, help="JSON file of question/expected page cases.")
    args = parser.parse_args()
    if not (args.wiki or args.epub or args.queries):
        parser.error("Choose --wiki, --epub, or --queries.")

    settings = Settings()
    if args.wiki or args.epub:
        documents: list[DocumentChunk] = []
        if args.wiki:
            documents.extend(await fetch_wiki(settings))
        if args.epub:
            documents.extend(fetch_epub(settings))
        report = analyze_documents(documents)
        issues = report.pop("issues")
        print(json.dumps(report, indent=2))
        print(f"\n{len(issues)} issue(s):")
        for issue in issues:
            print(f"- [{issue['page']}] {issue['issue']}")
        if args.show_page:
            for document in documents:
                if document.source_label == args.show_page:
                    for text, _ in chunk_document(document.source_label, document.text):
                        print(f"\n----- {len(text)} chars -----\n{text}")

    if args.queries:
        cases = json.loads(args.queries.read_text(encoding="utf-8"))
        results = evaluate_queries(KnowledgeBase(settings), cases)
        hits = 0
        for result in results:
            hit = result["rank"] is not None
            hits += hit
            status = f"rank {result['rank']}" if hit else "MISS"
            print(f"[{status}] {result['question']}\n    expected: {result['expected']}\n    top: {result['top']}")
        print(f"\nHit rate: {hits}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
