"""CLI entrypoint for the Hospital FIAP clinical assistant."""

from __future__ import annotations

import argparse
import json
import sys

from assistant.chains import run_assistant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hospital FIAP clinical assistant — LangChain pipeline"
    )
    parser.add_argument("--patient", required=True, help="Patient ID (e.g. PAC-001)")
    parser.add_argument("--query", required=True, help="Clinical question")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON instead of plain text",
    )
    args = parser.parse_args(argv)

    result = run_assistant(args.patient, args.query)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["response"])
        if result["sources"]:
            print("Fontes:", ", ".join(result["sources"]))
        if result["requires_human_validation"]:
            print("Requer validação humana: sim")
        if result["flags"]:
            print("Flags:", ", ".join(result["flags"]))

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
