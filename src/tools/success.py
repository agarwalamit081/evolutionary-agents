"""Per-tool success-contract evaluation (#11).

A tool's ``success_contract`` declares how to tell a REAL success from a
handler that returned WITHOUT raising but produced an error/empty surface.
The canonical case: ``git_clone`` returns ``"ERROR: <reason>"`` on failure
rather than raising, so — without a contract — it was recorded as
``success=True`` in ``tool_call_metrics`` (which feeds governance retirement +
the E2 selection blend). The contract makes that recorded signal honest.

Contract shape (additive; default none = today's behavior). Lives in
``TOOL_ANNOTATIONS[<tool>]["success_contract"]``::

    {
        "mode": "nonempty",                          # stripped output non-empty
        "exclude_prefixes": ["ERROR:", "DISABLED"],  # output starting with ⇒ fail
        "regex": r"\\b\\d+ files?\\b",                  # optional: output must match
    }

All present clauses must hold (AND). ``evaluate_success`` is PURE (no I/O, no
settings read) so the unit suite pins it without a registry/gateway. The
execute node reads the contract off the registry (``get_success_contract``)
and the ``TOOL_SUCCESS_CONTRACT_ENABLED`` flag (``AgentSettings``); on any
config/eval error it returns True (fail-open) so a malformed contract can
NEVER break a tool call.
"""

from __future__ import annotations

import re
from typing import Any

# A success contract is a small dict; the keys are well-known (see module
# docstring). Typed as a plain dict so it round-trips through TOOL_ANNOTATIONS
# (``dict[str, object]``) and the registry without a Pydantic model.
SuccessContract = dict[str, Any]


def evaluate_success(contract: SuccessContract | None, output: Any) -> bool:
    """Return whether ``output`` satisfies ``contract``.

    No contract (``None`` / empty) ⇒ ``True`` — today's behavior, where a
    non-raising handler is a success. Otherwise ALL present clauses must hold:

    * ``mode == "nonempty"`` ⇒ the stripped output is non-empty;
    * ``exclude_prefixes`` ⇒ the stripped output does NOT start with any listed
      prefix (case-sensitive — the canonical error surfaces ``"ERROR:"`` /
      ``"DISABLED:"`` are upper-case, and a tool that lower-cases its error
      banner should add the matching prefix);
    * ``regex`` ⇒ the stripped output matches the pattern via ``re.search``
      (partial match, not fullmatch).

    Args:
        contract: The tool's success contract, or ``None``/empty for "always
            success" (no contract).
        output: The tool's raw result (coerced to ``str`` if not already).

    Returns:
        ``True`` if the output satisfies the contract (or there is none).
    """
    if not contract:
        return True
    text = output.strip() if isinstance(output, str) else str(output).strip()

    if contract.get("mode") == "nonempty" and not text:
        return False

    for prefix in contract.get("exclude_prefixes") or []:
        if isinstance(prefix, str) and prefix and text.startswith(prefix):
            return False

    pattern = contract.get("regex")
    if isinstance(pattern, str) and pattern:
        try:
            if not re.search(pattern, text):
                return False
        except re.error:
            # A malformed contract pattern must never break a tool call — treat
            # it as "no regex clause" (the other clauses still apply).
            pass

    return True
