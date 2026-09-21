"""Triage one security report with jev via the TypeSafe SDK.

Put TYPESAFE_API_KEY=... in .env and run with:

    uv run --env-file .env jev-sec-triage/jev-triage.py
"""

import json

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

REPORT = (
    "/api/export?path=../../etc/passwd returns the file contents "
    "for any logged-in user. Reproduced on prod today."
)

QUESTIONS = {
    "is_security_report": Noul(instructions="Is this reporting a security vulnerability?"),
    "category": Choice(
        instructions="Which vulnerability class fits best?",
        criteria={
            "injection": "SQL, command, or template injection",
            "path_traversal": "Reading files outside the intended directory",
            "auth": "Authentication or authorization bypass",
            "xss": "Cross-site scripting",
            "other": "Anything else",
        },
    ),
    "severity": Score(
        instructions="How severe is the issue?",
        criteria=["informational", "low", "medium", "high", "critical"],
    ),
}


def main() -> None:
    with TypeSafeClient() as client:
        response = client.system_one(state=REPORT, questions=QUESTIONS)

    print(json.dumps(response.model_dump()["answers"], indent=2))
    print()
    print(f"model={response.model}")
    print(f"is_security_report p={response.nouls['is_security_report'].noul:.2f}")
    category = response.choices["category"]
    print(f"category={category.choice} confidence={category.confidence:.2f}")
    severity = response.scores["severity"]
    print(f"severity={severity.score:.2f} confidence={severity.confidence:.2f}")


if __name__ == "__main__":
    main()
