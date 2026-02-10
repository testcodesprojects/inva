"""Expression prior validation (muparser mini-language)."""

from __future__ import annotations

import re
from typing import Any

from .errors import SafetyError

# Functions recognised by the C binary's muparser evaluator.
_MUPARSER_FUNCTIONS = {
    "exp", "log", "log2", "log10", "sqrt", "abs",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "min", "max", "sum", "avg",
}

# Reserved constants / variables recognised by muparser + INLA.
_MUPARSER_RESERVED = {"pi", "theta", "return"}

# Characters that are safe inside expression prior strings.
_EXPRESSION_ALLOWED_CHARS = re.compile(
    r"^[a-zA-Z0-9_\s\+\-\*/\^();\.,=<>!]+$"
)


def _is_expression_prior(prior: Any) -> bool:
    """Return True if *prior* is an ``expression:`` prior string."""
    if not isinstance(prior, str):
        return False
    return prior.strip().lower().startswith("expression:")


def _validate_expression_prior(prior: str) -> None:
    """Validate an ``expression:`` prior string.

    Checks that:
    1. The body is non-empty.
    2. Only muparser-safe characters are used.
    3. A ``return(...)`` statement is present.
    4. No suspicious tokens (import, exec, eval, system, os, subprocess, etc.).
    """
    body = prior.strip()
    # Strip the "expression:" prefix
    idx = body.find(":")
    if idx < 0:
        raise SafetyError("pyinla safety check: expression prior must start with 'expression:'.")
    body = body[idx + 1:].strip()

    if not body:
        raise SafetyError("pyinla safety check: expression prior body is empty.")

    # Check for allowed characters only
    if not _EXPRESSION_ALLOWED_CHARS.match(body):
        bad_chars = set()
        for ch in body:
            if not _EXPRESSION_ALLOWED_CHARS.match(ch):
                bad_chars.add(repr(ch))
        raise SafetyError(
            "pyinla safety check: expression prior contains disallowed characters: {}.".format(
                ", ".join(sorted(bad_chars))
            )
        )

    # Must contain return(...)
    if "return(" not in body.replace(" ", ""):
        raise SafetyError(
            "pyinla safety check: expression prior must contain a return() statement."
        )

    # Block dangerous tokens (defence in depth — these are just strings to the
    # C binary, but we reject anything that looks like a code-injection attempt).
    body_lower = body.lower()
    _BLOCKED_TOKENS = {
        "import", "__", "exec", "eval", "system", "subprocess",
        "os.", "open(", "file(", "compile", "globals", "locals",
        "getattr", "setattr", "delattr",
    }
    for token in _BLOCKED_TOKENS:
        if token in body_lower:
            raise SafetyError(
                "pyinla safety check: expression prior contains blocked token '{}'.".format(token)
            )
