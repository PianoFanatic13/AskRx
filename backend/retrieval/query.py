import argparse

from backend.retrieval.hybrid import resolve_and_search


def _print_results(result: dict) -> None:
    print(f"filter_applied={result['filter_applied']}  match_type={result['match_type']}")
    if result["resolution_note"]:
        print(f"note: {result['resolution_note']}")
    if result["candidates"]:
        candidates = ", ".join(f"{c['name']} ({c['rxcui']})" for c in result["candidates"])
        print(f"candidates: {candidates}")
    print()

    if not result["results"]:
        print("(no results)")
        return

    for i, chunk in enumerate(result["results"], start=1):
        print(f"[{i}] rrf_score={chunk['rrf_score']:.5f}  {chunk['drug_name']}  rxcui={chunk['rxcui']}")
        print(f"    loinc_code={chunk['loinc_code']}  section={chunk['section_title_path']}")
        print(f"    {chunk['chunk_text'][:200]}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a query through hybrid retrieval and print the top results."
    )
    parser.add_argument("query_text", help="The natural-language query")
    parser.add_argument("--drug-name", default=None, help="Drug name to resolve and filter by")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--dsn", default=None, help="PostgreSQL DSN override")
    args = parser.parse_args()

    kwargs = {"top_k": args.top_k}
    if args.dsn is not None:
        kwargs["dsn"] = args.dsn

    result = resolve_and_search(args.query_text, args.drug_name, **kwargs)
    _print_results(result)


if __name__ == "__main__":
    main()
