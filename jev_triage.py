import json
import os
import urllib.request

req = urllib.request.Request(
    "https://ai-gateway.vercel.sh/v1/evaluate",
    headers={
        "Authorization": f"Bearer {os.environ['AI_GATEWAY_API_KEY']}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "typesafe-ai/jev",
        "state": "/api/export?path=../../etc/passwd returns the file contents "
                 "for any logged-in user. Reproduced on prod today.",
        "questions": {
            "is_security_report": {
                "type": "boolean",
                "instructions": "Is this reporting a security vulnerability?",
            },
            "category": {
                "type": "choice",
                "instructions": "Which vulnerability class fits best?",
                "criteria": {
                    "injection": "SQL, command, or template injection",
                    "path_traversal": "Reading files outside the intended directory",
                    "auth": "Authentication or authorization bypass",
                    "xss": "Cross-site scripting",
                    "other": "Anything else",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is the issue?",
                "criteria": ["informational", "low", "medium", "high", "critical"],
            },
        },
    }).encode(),
)

result = json.load(urllib.request.urlopen(req))
print(json.dumps(result["answers"], indent=2))
