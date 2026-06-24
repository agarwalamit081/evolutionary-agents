"""Typed correctness checks for the evaluation harness (Phase 3).

Each evaluator implements::

    async check(config, deliverables, state, *, gateway=None) -> CheckResult

``StructuralCheck`` / ``GoldenCheck`` are stdlib-only (csv/json) and run in any
env. ``ExecutionCheck`` runs a read-only invariant probe in the (subprocess)
sandbox. ``OracleCheck`` is the LLM-judge: it prefers deepeval/ragas when the
``[eval]`` extras import cleanly, falls back to a single structured gateway call,
and otherwise **skips** (no signal) so a missing optional dep can never block a
run. Skipped checks are excluded from the mean score and never gate completion.

Deliverable paths are resolved with the same resolver the verify node uses
(``results_root`` → ``workspace_root`` → literal), so a check and verify never
disagree on whether a deliverable exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.eval.models import CheckConfig, CheckResult, CorrectnessResult, GoalSpec
from src.tools._paths import (
    _subdir_active,
    get_active_run_id,
    resolve_existing,
    results_root,
    run_subdir_path,
    strip_results_prefix,
)

if TYPE_CHECKING:
    from src.graph.state import AgentState
    from src.llm.gateway import LLMGateway


# ─── Deliverable resolution (mirrors verify._resolve_deliverable) ─────────


def _resolve_deliverable(raw: str) -> Path | None:
    """Resolve a declared deliverable to its on-disk path, or ``None``.

    Checks ``results_root`` then ``workspace_root`` then a literal path,
    returning the first existing match. Kept in sync with
    ``src/graph/nodes/verify._resolve_deliverable`` so the eval layer and the
    verify node never disagree on deliverable existence.
    """
    candidates: list[Path] = []
    for base in ("results", "workspace"):
        try:
            candidates.append(resolve_existing(raw, base=base))
        except ValueError:
            continue
    parts = strip_results_prefix(Path(raw).parts)
    if parts:
        candidates.append(Path(*parts))
    candidates.append(Path(raw))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            if candidate.exists():
                resolved = candidate.resolve()
                key = str(resolved)
                if key not in seen:
                    seen.add(key)
                    return resolved
        except OSError:
            continue
    return None


def _effective_results_root() -> Path:
    """Results root the Execution/Golden probes should search.

    The probes ``os.walk(_RESULTS_ROOT)`` to locate deliverables the agent did
    NOT declare by name (e.g. q01's ``raw_events.jsonl``). That walk MUST be
    scoped to the *current run's* results cell — otherwise a canary/worker run
    that isolates its WRITES under ``results/<run_id>/`` (``set_active_run_id``)
    still has its READS scan the shared flat root, where a stale deliverable
    from another run lingers; the probe then cross-references that stale file
    against this run's own deliverables (q01 timestamp probe: fresh
    ``raw_events.jsonl`` vs a prior run's ``normalized.csv`` -> ``checked == 0``).
    Returns the run subdir when per-run isolation is active (a run_id is bound
    AND ``results_per_run_subdir`` is on), else the flat shared root. An unsafe
    run_id falls back to the flat root — never raises (the probe treats
    ``_RESULTS_ROOT`` as advisory).
    """
    run_id = get_active_run_id()
    if run_id:
        try:
            if _subdir_active():
                return run_subdir_path(run_id)
        except ValueError:
            pass  # unsafe run_id -> flat fallback (probes stay advisory)
    return results_root()


def _select_target(target: str | None, deliverables: list[str]) -> Path | None:
    """Pick the on-disk deliverable a check operates on.

    If ``target`` names a path, resolve it directly. Otherwise return the first
    present deliverable file (directories skipped — a check reads file content).
    """
    if target:
        resolved = _resolve_deliverable(target)
        if resolved is not None and resolved.is_file():
            return resolved
        return None
    for raw in deliverables:
        resolved = _resolve_deliverable(raw)
        if resolved is not None and resolved.is_file():
            return resolved
    return None


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"csv", "tsv"}:
        return "csv"
    if suffix == "jsonl":
        return "jsonl"
    if suffix == "json":
        return "json"
    return "text"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[unreadable: {exc}]"


def _load_struct(
    path: Path, fmt: str
) -> tuple[list[Any], list[str]]:
    """Parse a deliverable into (rows, field_names).

    CSV → list of dict rows + header. JSONL → list of decoded rows + the union
    of keys. JSON → a list (or a single dict wrapped in a list) + top-level keys.
    Text/unknown → lines as rows, no fields.
    """
    text = _read_text(path)
    if fmt == "csv":
        reader = csv.DictReader(text.splitlines())
        rows: list[Any] = list(reader)
        header: list[str] = list(reader.fieldnames or [])
        return rows, header
    if fmt == "jsonl":
        rows = []
        field_set: set[str] = set()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
            if isinstance(obj, dict):
                field_set.update(obj.keys())
        return rows, sorted(field_set)
    if fmt == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if isinstance(data, list):
            field_union: set[str] = set()
            for item in data:
                if isinstance(item, dict):
                    field_union.update(item.keys())
            return data, sorted(field_union)
        if isinstance(data, dict):
            return [data], sorted(data.keys())
        return [data], []
    # text / unknown
    return [ln for ln in text.splitlines() if ln.strip()], []


def _skipped(check_name: str, check_type: str, reason: str) -> CheckResult:
    """A check that carried no signal (optional dep absent, nothing to judge)."""
    return CheckResult(
        check_name=check_name,
        check_type=check_type,
        passed=True,
        score=1.0,
        evidence={"skipped": True, "reason": reason},
        skipped=True,
    )


# ─── Check base + implementations ────────────────────────────────────────


class CorrectnessCheck(ABC):
    """Base class for a single correctness evaluator."""

    name: str = "check"

    @abstractmethod
    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        """Run this check and return a CheckResult."""


class StructuralCheck(CorrectnessCheck):
    """Schema/keys/row-count validation for CSV/JSON/JSONL deliverables.

    Params: ``deliverable`` (optional target path), ``format`` (auto-detected),
    ``required_fields``, ``min_rows``, ``max_rows``, ``exact_rows``. The score is
    the fraction of declared conditions satisfied; passed requires all of them.
    """

    name = "structural"

    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        params = config.params
        resolved = _select_target(params.get("deliverable"), deliverables)
        if resolved is None:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                evidence={"reason": "target deliverable not on disk"},
            )

        fmt = params.get("format") or _detect_format(resolved)
        try:
            rows, field_names = _load_struct(resolved, fmt)
        except (ValueError, OSError) as exc:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                error=f"parse failed: {exc}",
            )

        row_count = len(rows) if isinstance(rows, list) else 1
        conditions: list[tuple[str, bool, dict[str, Any]]] = []

        required = params.get("required_fields") or []
        if required:
            present = set(field_names)
            missing = [f for f in required if f not in present]
            conditions.append(("required_fields", not missing, {"missing": missing}))

        for label, bound in (
            ("min_rows", params.get("min_rows")),
            ("max_rows", params.get("max_rows")),
            ("exact_rows", params.get("exact_rows")),
        ):
            if bound is None:
                continue
            if label == "min_rows":
                conditions.append((label, row_count >= int(bound), {"rows": row_count, "min": int(bound)}))
            elif label == "max_rows":
                conditions.append((label, row_count <= int(bound), {"rows": row_count, "max": int(bound)}))
            else:
                conditions.append((label, row_count == int(bound), {"rows": row_count, "expected": int(bound)}))

        if not conditions:
            conditions.append(("parsed_nonempty", row_count >= 1, {"rows": row_count}))

        passed = all(c[1] for c in conditions)
        score = sum(1 for c in conditions if c[1]) / len(conditions)
        return CheckResult(
            check_name=config.name,
            check_type=self.name,
            passed=passed,
            score=score,
            evidence={
                "format": fmt,
                "rows": row_count,
                "fields": field_names[:25],
                "conditions": {c[0]: c[2] for c in conditions},
            },
        )


class GoldenCheck(CorrectnessCheck):
    """Content assertions against a deliverable.

    Params: ``assertions`` — a list of dicts with ``kind`` in
    {exists, contains, regex, min_rows, max_rows, json_path_eq} plus fields:
    ``value`` (str/int), ``pattern`` (regex), ``path`` (dotted json key),
    ``tolerance`` (numeric), ``deliverable`` (optional target). The score is the
    fraction of assertions that hold.
    """

    name = "golden"

    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        assertions = config.params.get("assertions") or []
        if not assertions:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=True,
                score=1.0,
                evidence={"note": "no assertions declared"},
            )

        outcomes: list[tuple[str, bool, dict[str, Any]]] = []
        for assertion in assertions:
            kind = str(assertion.get("kind", "")).lower()
            resolved = _select_target(assertion.get("deliverable"), deliverables)
            ok, detail = self._eval_assertion(kind, assertion, resolved)
            outcomes.append((kind, ok, detail))

        passed = all(o[1] for o in outcomes)
        score = sum(1 for o in outcomes if o[1]) / len(outcomes)
        return CheckResult(
            check_name=config.name,
            check_type=self.name,
            passed=passed,
            score=score,
            evidence={"assertions": [{"kind": o[0], "passed": o[1], **o[2]} for o in outcomes]},
        )

    @staticmethod
    def _eval_assertion(
        kind: str, assertion: dict[str, Any], resolved: Path | None
    ) -> tuple[bool, dict[str, Any]]:
        if kind == "exists":
            ok = resolved is not None and resolved.exists()
            return ok, {"path": str(resolved) if resolved else None}
        if resolved is None:
            return False, {"reason": "deliverable not on disk"}

        if kind == "contains":
            text = _read_text(resolved)
            value = str(assertion.get("value", ""))
            return value in text, {"snippet_len": len(text)}
        if kind == "regex":
            text = _read_text(resolved)
            pattern = str(assertion.get("pattern", ""))
            try:
                return re.search(pattern, text) is not None, {"pattern": pattern}
            except re.error as exc:
                return False, {"pattern": pattern, "error": str(exc)}
        if kind in {"min_rows", "max_rows"}:
            rows, _ = _load_struct(resolved, _detect_format(resolved))
            n = len(rows) if isinstance(rows, list) else 1
            bound = int(assertion.get("value", 0))
            ok = n >= bound if kind == "min_rows" else n <= bound
            return ok, {"rows": n, "bound": bound}
        if kind == "json_path_eq":
            try:
                data = json.loads(_read_text(resolved))
            except json.JSONDecodeError as exc:
                return False, {"error": f"invalid json: {exc}"}
            target_val = _dotted_get(data, str(assertion.get("path", "")))
            expected = assertion.get("value")
            tolerance = assertion.get("tolerance")
            if tolerance is not None and isinstance(target_val, (int, float)) and isinstance(expected, (int, float)):
                return abs(target_val - expected) <= float(tolerance), {
                    "path": assertion.get("path"),
                    "got": target_val,
                    "expected": expected,
                    "tolerance": tolerance,
                }
            return target_val == expected, {
                "path": assertion.get("path"),
                "got": target_val,
                "expected": expected,
            }
        return False, {"reason": f"unknown assertion kind '{kind}'"}


def _dotted_get(data: Any, path: str) -> Any:
    """Traverse a dotted/``a.b.c`` path into nested dicts/lists."""
    cur: Any = data
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class ExecutionCheck(CorrectnessCheck):
    """Run a read-only invariant probe in the subprocess sandbox.

    Params: ``code`` (Python snippet) and optional ``timeout`` (default 30s).
    The snippet is prefixed with ``_DELIVERABLES`` (a JSON list of resolved
    absolute deliverable paths) and ``_RESULTS_ROOT`` so it can read the agent's
    outputs. The probe passes iff it exits 0. Always subprocess mode — the
    deliverables are local files the agent just wrote; docker isolation is
    unnecessary and its read-only fs would hide them.
    """

    name = "execution"

    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        code = config.params.get("code")
        if not code:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                error="execution check requires a 'code' probe",
            )

        resolved_paths = [str(p) for p in map(_resolve_deliverable, deliverables) if p]
        results_root = ""
        try:
            results_root = str(_effective_results_root())
        except Exception:  # noqa: BLE001 — results_root is advisory for the probe
            results_root = ""

        # Double-encoded JSON → a safe Python string literal of the JSON list.
        deliverables_literal = json.dumps(json.dumps(resolved_paths))
        preamble = (
            "import json as _json\n"
            f"_DELIVERABLES = _json.loads({deliverables_literal})\n"
            f"_RESULTS_ROOT = {json.dumps(results_root)}\n"
        )
        full_script = preamble + "\n" + str(code)

        # subprocess mode only (see class docstring).
        from src.sandbox.executor import SandboxExecutor

        timeout = int(config.params.get("timeout", 30))
        sandbox = SandboxExecutor(SimpleNamespace(evolution_sandbox_mode="subprocess"))
        result = await sandbox.execute_code(full_script, timeout=timeout)

        passed = bool(result.success and result.exit_code == 0)
        return CheckResult(
            check_name=config.name,
            check_type=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            evidence={
                "exit_code": result.exit_code,
                "duration_s": result.duration_seconds,
                "timed_out": result.timed_out,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-1000:],
            },
        )


class IdempotencyCheck(CorrectnessCheck):
    """Prove a transform is deterministic by running it twice and comparing.

    Params: ``transform_code`` (a Python snippet; alias ``code`` accepted),
    ``input_deliverable`` (optional — which declared deliverable to feed as
    ``_INPUT``; defaults to the first file deliverable), ``timeout`` (default
    30s). The snippet runs TWICE in the subprocess sandbox with ``_INPUT`` (abs
    path to the input deliverable), ``_DELIVERABLES`` and ``_RESULTS_ROOT`` in
    scope; it should ``print()`` its deterministic result. The check PASSES iff
    both runs exit 0 AND produce byte-identical, non-empty stdout — i.e. the
    transform is reproducible. This is the determinism methodology: a transform
    whose output drifts between identical inputs (wall-clock, unseeded RNG,
    unordered-dict iteration) is caught here, deterministically, without trusting
    the agent's own ``is_idempotent`` claim (cf. the q06 manifest cross-check).
    """

    name = "idempotency"

    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        code = config.params.get("transform_code") or config.params.get("code")
        if not code:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                error="idempotency check requires a 'transform_code' probe",
            )

        # Resolve the input deliverable the transform reads as _INPUT.
        input_target = config.params.get("input_deliverable")
        input_path: Path | None = None
        if input_target:
            resolved = _resolve_deliverable(input_target)
            if resolved is not None and resolved.is_file():
                input_path = resolved
        else:
            for raw in deliverables:
                resolved = _resolve_deliverable(raw)
                if resolved is not None and resolved.is_file():
                    input_path = resolved
                    break
        if input_path is None:
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                evidence={"reason": "input deliverable not on disk"},
            )

        resolved_paths = [str(p) for p in map(_resolve_deliverable, deliverables) if p]
        results_root = ""
        try:
            results_root = str(_effective_results_root())
        except Exception:  # noqa: BLE001 — results_root is advisory for the probe
            results_root = ""

        # Same double-encoded-JSON preamble shape as ExecutionCheck, plus _INPUT.
        deliverables_literal = json.dumps(json.dumps(resolved_paths))
        preamble = (
            "import json as _json\n"
            f"_DELIVERABLES = _json.loads({deliverables_literal})\n"
            f"_RESULTS_ROOT = {json.dumps(results_root)}\n"
            f"_INPUT = {json.dumps(str(input_path))}\n"
        )
        full_script = preamble + "\n" + str(code)

        from src.sandbox.executor import SandboxExecutor

        timeout = int(config.params.get("timeout", 30))
        sandbox = SandboxExecutor(SimpleNamespace(evolution_sandbox_mode="subprocess"))
        run1 = await sandbox.execute_code(full_script, timeout=timeout)
        run2 = await sandbox.execute_code(full_script, timeout=timeout)

        out1 = (getattr(run1, "stdout", "") or "").strip()
        out2 = (getattr(run2, "stdout", "") or "").strip()
        both_ok = bool(
            run1.success
            and run1.exit_code == 0
            and run2.success
            and run2.exit_code == 0
        )
        identical = bool(both_ok and out1 != "" and out1 == out2)

        conditions: list[tuple[str, bool]] = [
            ("both_runs_exit_zero", both_ok),
            ("outputs_identical", identical),
        ]
        passed = all(c[1] for c in conditions)
        score = sum(1 for c in conditions if c[1]) / len(conditions)
        return CheckResult(
            check_name=config.name,
            check_type=self.name,
            passed=passed,
            score=score,
            evidence={
                "exit_codes": [run1.exit_code, run2.exit_code],
                "output_sha256": [_sha256(out1), _sha256(out2)],
                "identical": identical,
                "run1_stdout_tail": out1[-1000:],
                "run2_stderr_tail": (getattr(run2, "stderr", "") or "")[-500:],
            },
        )


def _sha256(text: str) -> str:
    """Stable hex digest of a probe's captured stdout (for evidence/logging)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class OracleCheck(CorrectnessCheck):
    """LLM-judge over the deliverable vs the goal/reference.

    Prefers deepeval (``GEval``) and ragas (faithfulness/answer-relevancy) when
    the optional ``[eval]`` extras import cleanly; falls back to a single
    structured gateway call (faithfulness + answer-relevancy, 0–1) so a battery
    run still gets a real signal when ragas is broken upstream. Skips (no
    signal) when the judge is disabled or no deliverable exists.
    """

    name = "oracle"

    async def check(
        self,
        config: CheckConfig,
        deliverables: list[str],
        state: AgentState,
        *,
        gateway: LLMGateway | None = None,
    ) -> CheckResult:
        from src.config.settings import get_settings

        eval_settings = get_settings().eval
        if not eval_settings.eval_llm_judge_enabled:
            return _skipped(config.name, self.name, "LLM judge disabled (EVAL_LLM_JUDGE_ENABLED=false)")

        resolved = _select_target(config.params.get("deliverable"), deliverables)
        if resolved is None:
            return _skipped(config.name, self.name, "no deliverable on disk to judge")

        reference = str(config.params.get("reference", ""))
        text = _read_text(resolved)
        if not text.strip():
            return CheckResult(
                check_name=config.name,
                check_type=self.name,
                passed=False,
                score=0.0,
                evidence={"reason": "deliverable empty"},
            )

        score, method, detail = await self._judge(text, reference, gateway)
        if method == "skipped":
            return _skipped(config.name, self.name, str(detail.get("reason", "judge unavailable")))
        passed = score >= eval_settings.eval_canary_min_score
        return CheckResult(
            check_name=config.name,
            check_type=self.name,
            passed=passed,
            score=score,
            evidence={"method": method, **detail},
        )

    async def _judge(
        self, text: str, reference: str, gateway: LLMGateway | None
    ) -> tuple[float, str, dict[str, Any]]:
        """Return (score 0–1, method label, evidence). method=='skipped' → none."""
        # 1. deepeval (best-effort; lazy).
        scored = await self._judge_deepeval(text, reference)
        if scored is not None:
            return scored[0], "deepeval", scored[1]
        # 2. ragas (best-effort; lazy).
        scored = await self._judge_ragas(text, reference)
        if scored is not None:
            return scored[0], "ragas", scored[1]
        # 3. gateway structured judge (the reliable battery-run path).
        if gateway is not None:
            scored = await self._judge_gateway(gateway, text, reference)
            if scored is not None:
                return scored[0], "gateway-llm-judge", scored[1]
        return 0.0, "skipped", {"reason": "no judge available (deepeval/ragas absent and no gateway)"}

    @staticmethod
    async def _judge_deepeval(
        text: str, reference: str
    ) -> tuple[float, dict[str, Any]] | None:
        """GEval faithfulness via deepeval; None if unavailable/unable."""
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase
            from deepeval.test_case import LLMTestCaseParams
        except Exception as exc:  # noqa: BLE001 — optional dep
            logger.debug("deepeval judge unavailable: {}", exc)
            return None

        # deepeval's enum/class shapes drift across versions (LLMTestCaseParams
        # is sometimes aliased to MultiTurnParams); treat the API surface as Any
        # so a version drift can't break type-checking of this best-effort path
        # (any runtime error is caught below).
        g_eval_cls: Any = GEval
        params_enum: Any = LLMTestCaseParams
        try:
            test_case: Any = LLMTestCase(input=reference or "goal", actual_output=text)
            metric: Any = g_eval_cls(
                name="faithfulness",
                criteria="Is every claim in the output supported by the goal/reference?",
                evaluation_params=[params_enum.INPUT, params_enum.ACTUAL_OUTPUT],
            )
            metric.measure(test_case)
            score = max(0.0, min(1.0, float(metric.score or 0.0)))
            return score, {"faithfulness": score, "reason": str(getattr(metric, "reason", ""))}
        except Exception as exc:  # noqa: BLE001 — never let the judge crash the run
            logger.debug("deepeval judge failed: {}", exc)
            return None

    @staticmethod
    async def _judge_ragas(
        text: str, reference: str
    ) -> tuple[float, dict[str, Any]] | None:
        """ragas faithfulness; None if unavailable/unable (e.g. broken mistralai)."""
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy
        except Exception as exc:  # noqa: BLE001 — optional/broken dep
            logger.debug("ragas judge unavailable: {}", exc)
            return None

        try:
            ds: Any = Dataset.from_dict(
                {"question": [reference or "goal"], "answer": [text], "contexts": [[reference]]}
            )
            # ragas evaluate() returns EvaluationResult (not a plain dict) and
            # its accessors vary by version — treat as Any here.
            result: Any = ragas_evaluate(ds, metrics=[faithfulness, answer_relevancy])
            faith = float(result.get("faithfulness", 0.0) or 0.0)
            rel = float(result.get("answer_relevancy", 0.0) or 0.0)
            score = max(0.0, min(1.0, (faith + rel) / 2))
            return score, {"faithfulness": faith, "answer_relevancy": rel}
        except Exception as exc:  # noqa: BLE001
            logger.debug("ragas judge failed: {}", exc)
            return None

    @staticmethod
    async def _judge_gateway(
        gateway: LLMGateway, text: str, reference: str
    ) -> tuple[float, dict[str, Any]] | None:
        """Structured single-call judge (faithfulness + answer-relevancy, 0–1)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict correctness judge. Score the DELIVERABLE "
                    "against the GOAL on two axes, each 0.0–1.0:\n"
                    "- faithfulness: every claim is supported by / consistent with the goal\n"
                    "- answer_relevancy: the deliverable addresses what the goal asked for\n"
                    "Respond with ONLY JSON: "
                    '{"faithfulness": <float>, "answer_relevancy": <float>, "reason": "<short>"}'
                ),
            },
            {
                "role": "user",
                "content": f"GOAL:\n{reference or '(none)'}\n\nDELIVERABLE:\n{text[:8000]}",
            },
        ]
        try:
            resp = await gateway.acompletion(messages, temperature=0.0)
        except Exception as exc:  # noqa: BLE001 — judge must never crash the run
            logger.debug("gateway judge call failed: {}", exc)
            return None

        data = _parse_json_lenient(resp.content)
        if not isinstance(data, dict):
            return None
        try:
            faith = float(data.get("faithfulness", 0.0))
            rel = float(data.get("answer_relevancy", 0.0))
        except (TypeError, ValueError):
            return None
        faith = max(0.0, min(1.0, faith))
        rel = max(0.0, min(1.0, rel))
        score = (faith + rel) / 2
        return score, {
            "faithfulness": faith,
            "answer_relevancy": rel,
            "reason": str(data.get("reason", ""))[:300],
            "model": getattr(resp, "model", ""),
        }


def _parse_json_lenient(content: str) -> Any:
    """Parse JSON from an LLM response, salvaging malformed JSON via json-repair."""
    text = (content or "").strip()
    # Strip a ```json fence if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            return json.loads(repair_json(text))
        except Exception:  # noqa: BLE001 — last resort
            return None


# ─── Registry + runner ───────────────────────────────────────────────────


CHECK_REGISTRY: dict[str, CorrectnessCheck] = {
    "structural": StructuralCheck(),
    "golden": GoldenCheck(),
    "execution": ExecutionCheck(),
    "idempotency": IdempotencyCheck(),
    "oracle": OracleCheck(),
}


async def run_checks(
    spec: GoalSpec,
    deliverables: list[str],
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
) -> CorrectnessResult:
    """Run every check declared on ``spec`` and aggregate the outcomes.

    Skipped checks (no signal) are excluded from both the mean score and the
    pass gate. A spec with no checks is a trivial pass (score 1.0) — it signals
    "this goal has no machine-verifiable criteria", not a failure.
    """
    if not spec.checks:
        return CorrectnessResult(spec_id=spec.spec_id, overall_score=1.0, passed=True, checks=[])

    # A correctness check must be able to see EVERY deliverable the spec names,
    # not merely the ones the agent self-reported. Execution probes resolve their
    # inputs from this list (``_DELIVERABLES``); if the agent omits a spec-named
    # deliverable from its declared paths, the probe for that file becomes
    # vacuously unresolvable and fails on otherwise-correct output (battery-04
    # q05 scored 0.62 on correct data because the agent declared reconciled.csv
    # but not integrity_report.json/audit.json). The spec's expected deliverables
    # take precedence (resolved first); the run's declared paths supplement them.
    merged: list[str] = []
    _seen: set[str] = set()
    for _raw in (*spec.expected_deliverables, *deliverables):
        if _raw not in _seen:
            _seen.add(_raw)
            merged.append(_raw)

    results: list[CheckResult] = []
    for cfg in spec.checks:
        check = CHECK_REGISTRY.get(cfg.check_type)
        if check is None:
            results.append(
                CheckResult(
                    check_name=cfg.name,
                    check_type=cfg.check_type,
                    passed=False,
                    score=0.0,
                    error=f"unknown check type '{cfg.check_type}'",
                )
            )
            continue
        try:
            res = await check.check(cfg, merged, state, gateway=gateway)
        except Exception as exc:  # noqa: BLE001 — a crashing check must not abort verification
            logger.exception("Correctness check '{}' raised", cfg.name)
            res = CheckResult(
                check_name=cfg.name,
                check_type=cfg.check_type,
                passed=False,
                score=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(res)

    scored = [r for r in results if not r.skipped]
    overall = (sum(r.score for r in scored) / len(scored)) if scored else 1.0
    passed = all(r.passed for r in scored) if scored else True
    return CorrectnessResult(spec_id=spec.spec_id, overall_score=overall, passed=passed, checks=results)
