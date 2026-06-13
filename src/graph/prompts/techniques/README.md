# Prompt-technique reference library

Curated, prompt-only technique templates (no graph/architecture changes).
The agent's node system-prompts in `../templates/` compose *principles* from
these; these files are the authoritative reference for each technique.

- `basic/`   — single-pass techniques (role prompting, step-back, structured CoT,
  least-to-most, self-ask, generated knowledge, checklist, negative prompting,
  certainty/uncertainty, system-2 attention, …).
- `looping/` — iterative techniques (reflection, chain-of-verification,
  thought-critique-improve, self-refine, …).

`branching/` (tree-of-thought, self-consistency voting, etc.) is intentionally
**not** copied: those require multi-call fan-out and graph changes, tracked as a
follow-up — see the design docs.
