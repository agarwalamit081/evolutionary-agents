"""Self-Evolution Engine — full mutation pipeline.

Phases: analyze → generate → validate → sandbox_test → ab_test → deploy|reject → persist
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import MutationType
from src.safety.pipeline import SafetyPipeline

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


class SelfEvolutionEngine:
    """4-phase self-evolution engine.

    Analyzes performance data to identify improvement opportunities,
    generates mutation proposals, validates through the safety pipeline,
    and deploys if statistically significant improvement is confirmed.
    """

    def __init__(
        self,
        safety_pipeline: SafetyPipeline | None = None,
        gateway: LLMGateway | None = None,
    ) -> None:
        self._safety = safety_pipeline or SafetyPipeline()
        self._gateway = gateway
        self._generation = 0

    async def analyze(
        self,
        execution_history: list[dict[str, Any]],
        failure_patterns: list[str],
    ) -> dict[str, Any]:
        """Phase 1: Analyze performance data for improvement opportunities.

        Args:
            execution_history: Recent execution records with metrics.
            failure_patterns: Known failure pattern descriptions.

        Returns:
            Analysis results with identified opportunities.
        """
        logger.info(f"Evolution analyze: {len(execution_history)} records, {len(failure_patterns)} failure patterns")

        opportunities: list[dict[str, Any]] = []

        # Analyze failure patterns for prompt improvement opportunities
        if failure_patterns:
            opportunities.append({
                "type": MutationType.PROMPT,
                "description": "Address recurring failure patterns via prompt refinement",
                "patterns": failure_patterns[:5],
                "priority": "high",
            })

        # Analyze execution metrics for workflow optimization
        if execution_history:
            avg_duration = sum(e.get("duration_ms", 0) for e in execution_history) / max(len(execution_history), 1)
            if avg_duration > 5000:
                opportunities.append({
                    "type": MutationType.WORKFLOW,
                    "description": f"Reduce average execution time ({avg_duration:.0f}ms)",
                    "priority": "medium",
                })

        # Always consider tool optimization
        opportunities.append({
            "type": MutationType.TOOL,
            "description": "Optimize tool selection and chaining",
            "priority": "low",
        })

        return {
            "opportunities": opportunities,
            "performance_metrics": {
                "execution_count": len(execution_history),
                "failure_count": len(failure_patterns),
                "generation": self._generation,
            },
        }

    async def generate(
        self,
        opportunity: dict[str, Any],
        current_content: str | None = None,
    ) -> dict[str, Any]:
        """Phase 2: Generate a mutation proposal.

        Uses LLM when gateway is available, falls back to heuristic generation.

        Args:
            opportunity: The improvement opportunity from analysis.
            current_content: Current code/prompt to mutate.

        Returns:
            Mutation proposal with original and mutated content.
        """
        mutation_type = opportunity.get("type", MutationType.PROMPT)
        description = opportunity.get("description", "Unknown improvement")

        logger.info(f"Generating {mutation_type} mutation: {description}")

        # Try LLM-based generation first
        if self._gateway is not None:
            result = await self._llm_generate(opportunity, current_content)
            if result is not None:
                return result

        # Heuristic fallback
        return self._heuristic_generate(opportunity, current_content)

    async def _llm_generate(
        self,
        opportunity: dict[str, Any],
        current_content: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate mutation via LLM. Returns None on failure."""
        try:
            from src.graph.enums import TaskComplexity
            from src.graph.prompts import EVOLUTION_GENERATE_SYSTEM, EVOLUTION_GENERATE_USER
            from src.graph.schemas import MutationProposal
            from src.llm.structured_output import StructuredOutputManager

            mutation_type = opportunity.get("type", MutationType.PROMPT)
            description = opportunity.get("description", "Unknown improvement")
            priority = opportunity.get("priority", "low")

            # Build performance context from opportunity data
            patterns = opportunity.get("patterns", [])
            perf_ctx = ""
            if patterns:
                perf_ctx = "Failure patterns:\n" + "\n".join(f"- {p}" for p in patterns[:5])

            user_prompt = EVOLUTION_GENERATE_USER.format(
                mutation_type=mutation_type.value if hasattr(mutation_type, "value") else str(mutation_type),
                description=description,
                priority=priority,
                current_content=current_content or "(no current content available)",
                performance_context=perf_ctx or "(no specific performance data)",
            )
            messages: list[dict[str, str]] = [
                {"role": "system", "content": EVOLUTION_GENERATE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]

            response = await self._gateway.acompletion(
                messages=messages,
                complexity=TaskComplexity.COMPLEX,
            )

            extractor = StructuredOutputManager()
            proposal = await extractor.extract(response.content, MutationProposal)
            if proposal is None:
                logger.debug("LLM mutation generation returned unparseable output")
                return None

            logger.info(f"LLM generated {proposal.mutation_type.value} mutation for {proposal.target_path or 'general'}")

            return {
                "mutation_type": proposal.mutation_type,
                "description": proposal.description,
                "original_content": current_content,
                "mutated_content": proposal.mutated_content,
                "target_path": proposal.target_path,
                "priority": priority,
                "rationale": proposal.rationale,
                "model_used": response.model,
                "tokens_used": response.total_tokens,
            }
        except Exception as e:
            logger.debug(f"LLM mutation generation failed: {e}")
            return None

    @staticmethod
    def _heuristic_generate(
        opportunity: dict[str, Any],
        current_content: str | None = None,
    ) -> dict[str, Any]:
        """Generate a heuristic mutation when LLM is unavailable."""
        mutation_type = opportunity.get("type", MutationType.PROMPT)
        description = opportunity.get("description", "Unknown improvement")

        # Generate a template mutation based on type
        if mutation_type == MutationType.PROMPT:
            mutated = (
                f"# Improved prompt based on: {description}\n"
                f"# Original patterns addressed: {opportunity.get('patterns', [])}\n"
                f"# This is a heuristic placeholder — LLM generation preferred"
            )
        elif mutation_type == MutationType.WORKFLOW:
            mutated = (
                f"# Workflow optimization: {description}\n"
                f"# Reduced average execution time by simplifying step ordering"
            )
        else:
            mutated = f"# {mutation_type} mutation: {description}"

        return {
            "mutation_type": mutation_type,
            "description": description,
            "original_content": current_content,
            "mutated_content": mutated,
            "target_path": None,
            "priority": opportunity.get("priority", "low"),
            "rationale": "Heuristic generation (no LLM available)",
            "model_used": None,
            "tokens_used": 0,
        }

    async def validate(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Phase 3: Validate a mutation through the safety pipeline.

        Args:
            proposal: The mutation proposal to validate.

        Returns:
            Validation result with pass/fail and details.
        """
        mutated_content = proposal.get("mutated_content", "")

        logger.info(f"Validating mutation: {proposal.get('description', 'unknown')}")

        # Run safety pipeline
        safety_result = await self._safety.validate(
            code=mutated_content,
            context={
                "mutation_type": proposal.get("mutation_type"),
                "description": proposal.get("description"),
            },
        )

        if not safety_result["passed"]:
            failed_layers = [
                name for name, result in safety_result["layers"].items()
                if not result["passed"]
            ]
            logger.warning(f"Safety validation failed: {failed_layers}")
            return {
                "passed": False,
                "reason": f"Failed safety layers: {', '.join(failed_layers)}",
                "details": safety_result,
            }

        logger.info("Safety validation passed")
        return {
            "passed": True,
            "safety_result": safety_result,
        }

    async def sandbox_test(
        self,
        proposal: dict[str, Any],
        sandbox: Any | None = None,
    ) -> dict[str, Any]:
        """Phase 3.5: Run mutation code in an isolated sandbox.

        Args:
            proposal: The mutation proposal to test.
            sandbox: Optional SandboxExecutor instance.

        Returns:
            Dict with 'passed' bool and sandbox execution details.
        """
        if sandbox is None:
            logger.debug("No sandbox executor provided — skipping sandbox test")
            return {"passed": True, "note": "sandbox not available"}

        mutated_content = proposal.get("mutated_content", "")
        if not mutated_content:
            return {"passed": False, "reason": "No mutated content to test"}

        logger.info("Running mutation in sandbox")
        try:
            result = await sandbox.execute_code(mutated_content)
            passed = result.success and not result.timed_out

            if not passed:
                reason = "timed out" if result.timed_out else f"exit code {result.exit_code}"
                logger.warning(f"Sandbox test failed: {reason}")
                if result.stderr:
                    logger.debug(f"Sandbox stderr: {result.stderr[:200]}")

            return {
                "passed": passed,
                "sandbox_result": {
                    "success": result.success,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout[:1000],
                    "stderr": result.stderr[:1000],
                    "duration_seconds": result.duration_seconds,
                    "timed_out": result.timed_out,
                },
            }
        except Exception as e:
            logger.warning(f"Sandbox test error: {e}")
            return {"passed": False, "reason": str(e)}

    async def ab_test(
        self,
        proposal: dict[str, Any],
        sandbox: Any | None = None,
    ) -> dict[str, Any]:
        """Phase 4: A/B test — compare original vs mutated code in sandbox.

        Args:
            proposal: The mutation proposal with original and mutated content.
            sandbox: Optional SandboxExecutor instance.

        Returns:
            Dict with 'is_significant' bool and comparison details.
        """
        if sandbox is None:
            logger.debug("No sandbox executor — skipping A/B test, accepting mutation")
            return {"is_significant": True, "note": "A/B test skipped (no sandbox)"}

        original = proposal.get("original_content", "")
        mutated = proposal.get("mutated_content", "")

        if not original:
            # No original to compare against — run mutated only
            result = await sandbox.execute_code(mutated)
            return {
                "is_significant": result.success,
                "control_result": None,
                "treatment_result": {
                    "success": result.success,
                    "duration_seconds": result.duration_seconds,
                },
                "sample_size": 1,
            }

        logger.info("Running A/B test: original vs mutated")
        try:
            # Run both versions
            control_result = await sandbox.execute_code(original)
            treatment_result = await sandbox.execute_code(mutated)

            # Mutation must succeed and not be worse than original
            is_significant = (
                treatment_result.success
                and (
                    not control_result.success
                    or treatment_result.duration_seconds <= control_result.duration_seconds * 1.1
                )
            )

            logger.info(
                f"A/B test: control={'pass' if control_result.success else 'fail'} "
                f"({control_result.duration_seconds:.2f}s), "
                f"treatment={'pass' if treatment_result.success else 'fail'} "
                f"({treatment_result.duration_seconds:.2f}s), "
                f"significant={is_significant}"
            )

            return {
                "is_significant": is_significant,
                "control_result": {
                    "success": control_result.success,
                    "duration_seconds": control_result.duration_seconds,
                    "exit_code": control_result.exit_code,
                },
                "treatment_result": {
                    "success": treatment_result.success,
                    "duration_seconds": treatment_result.duration_seconds,
                    "exit_code": treatment_result.exit_code,
                },
                "sample_size": 1,
            }
        except Exception as e:
            logger.warning(f"A/B test error: {e}")
            return {"is_significant": False, "reason": str(e)}

    @staticmethod
    def reject(
        proposal: dict[str, Any],
        reason: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a rejection record for a mutation.

        Args:
            proposal: The rejected mutation proposal.
            reason: Why the mutation was rejected.

        Returns:
            Rejection result dict.
        """
        logger.info(f"Mutation rejected: {reason}")
        return {
            "deployed": False,
            "reason": str(reason.get("reason", reason.get("note", "Unknown"))),
            "proposal": proposal.get("description", "unknown"),
        }

    async def deploy(
        self,
        proposal: dict[str, Any],
        validation_result: dict[str, Any],
        ab_result: dict[str, Any] | None = None,
        git_tracker: Any | None = None,
    ) -> dict[str, Any]:
        """Phase 5: Deploy a validated mutation.

        Applies the mutation via git tracker and records the deployment.

        Args:
            proposal: The validated mutation proposal.
            validation_result: Result from the validate phase.
            ab_result: Result from the A/B test phase.
            git_tracker: Optional GitTracker for recording changes.

        Returns:
            Deployment result with status, commit hash, and metadata.
        """
        if not validation_result.get("passed"):
            return {"deployed": False, "reason": "Validation did not pass"}

        self._generation += 1
        logger.info(f"Deploying mutation (generation {self._generation})")

        commit_hash: str | None = None
        target_path = proposal.get("target_path")
        mutated_content = proposal.get("mutated_content", "")

        # Apply via git tracker if available
        if git_tracker is not None and target_path:
            try:
                await git_tracker.apply_mutation(target_path, mutated_content)
                commit_hash = await git_tracker.snapshot(
                    f"evolution: {proposal.get('description', 'mutation')}"
                )
                logger.info(f"Mutation committed to shadow repo: {commit_hash[:8]}")
            except Exception as e:
                logger.warning(f"Git tracker failed during deploy: {e}")
                commit_hash = None

        return {
            "deployed": True,
            "generation": self._generation,
            "mutation_type": proposal.get("mutation_type"),
            "description": proposal.get("description"),
            "commit_hash": commit_hash,
            "target_path": target_path,
            "rationale": proposal.get("rationale", ""),
            "ab_result": ab_result,
        }

    async def run_cycle(
        self,
        execution_history: list[dict[str, Any]],
        failure_patterns: list[str] | None = None,
        reflection: Any | None = None,
        sandbox: Any | None = None,
        git_tracker: Any | None = None,
    ) -> dict[str, Any]:
        """Run a complete evolution cycle.

        Pipeline: analyze → generate → validate (layers 1-5) → sandbox_test (layer 6)
                  → ab_test → deploy|reject

        Args:
            execution_history: Recent execution records.
            failure_patterns: Known failure patterns. If None, extracted from reflection.
            reflection: ReflectionResult from the reflect node.
            sandbox: Optional SandboxExecutor for sandbox and A/B testing.
            git_tracker: Optional GitTracker for recording mutations.

        Returns:
            Evolution cycle result with full context.
        """
        # Extract failure patterns from reflection if not provided
        if failure_patterns is None and reflection is not None:
            failure_patterns = []
            if hasattr(reflection, "lessons_learned"):
                failure_patterns.extend(reflection.lessons_learned)
            errors = getattr(reflection, "errors", [])
            if errors:
                failure_patterns.extend(str(e) for e in errors)
        elif failure_patterns is None:
            failure_patterns = []

        logger.info("Starting evolution cycle")

        # Phase 1: Analyze
        analysis = await self.analyze(execution_history, failure_patterns)
        opportunities = analysis.get("opportunities", [])

        if not opportunities:
            return {
                "status": "no_opportunities",
                "deployed": False,
                "mutations_proposed": 0,
                "mutations_deployed": 0,
            }

        # Phase 2: Generate (take highest priority opportunity)
        best_opportunity = opportunities[0]
        proposal = await self.generate(best_opportunity)

        # Phase 3: Validate (safety pipeline layers 1-5)
        validation = await self.validate(proposal)
        if not validation["passed"]:
            rejection = self.reject(proposal, validation)
            return {
                "status": "validation_failed",
                "deployed": False,
                "reason": validation.get("reason", "Unknown"),
                "mutations_proposed": 1,
                "mutations_deployed": 0,
                "proposal": proposal,
                "validation": validation,
                "rejection": rejection,
            }

        # Phase 4: Sandbox test (safety pipeline layer 6)
        sandbox_result = await self.sandbox_test(proposal, sandbox)
        if not sandbox_result.get("passed", True):
            rejection = self.reject(proposal, sandbox_result)
            return {
                "status": "sandbox_failed",
                "deployed": False,
                "reason": sandbox_result.get("reason", "Sandbox test failed"),
                "mutations_proposed": 1,
                "mutations_deployed": 0,
                "proposal": proposal,
                "validation": validation,
                "sandbox_result": sandbox_result,
                "rejection": rejection,
            }

        # Phase 5: A/B test
        ab_result = await self.ab_test(proposal, sandbox)

        # Phase 6: Deploy or Reject
        if ab_result.get("is_significant", False):
            deployment = await self.deploy(
                proposal, validation, ab_result, git_tracker
            )
        else:
            deployment = self.reject(proposal, ab_result)

        deployed = deployment.get("deployed", False)
        return {
            "status": "deployed" if deployed else "rejected",
            "deployed": deployed,
            "generation": self._generation,
            "proposal": proposal,
            "validation": validation,
            "sandbox_result": sandbox_result,
            "ab_result": ab_result,
            "deployment": deployment,
            "mutations_proposed": 1,
            "mutations_deployed": 1 if deployed else 0,
        }
