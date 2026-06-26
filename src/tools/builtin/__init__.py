"""Built-in tools package — 22 distinct tools for the agent.

The original 14 tools were audited for true duplicates (M7b); the 2 corpus
tools (Phase 1 search stack) are additive — `index_corpus` (write) and
`corpus_search` (read) are a distinct gather/recall cluster, not a dup of
`web_search` (live) or `web_scraper` (single page). `create_scheduled_task`
(Phase 5 I1) is another distinct cluster — durable future-work scheduling, not
a compute/read/write of the present moment. `git_clone` + `code_search` (Phase 5
I2) are yet another — index an external REPO's source into semantic memory +
recall a symbol by query, distinct from corpus (web PAGES) and memory_search
(tier memory). All names and descriptions remain unique; the similar clusters
(listing / reading / fetching) are deliberately distinct, not mergeable.
``test_consolidation.py`` locks this in — it fails if a future change duplicates
a name/description or collapses a cluster, preventing the B3 capability bloat
that dynamic-tool dedup addresses.
"""

from src.tools.builtin.arxiv_search import TOOL_DEFINITION as ARXIV_SEARCH_DEF
from src.tools.builtin.code_executor import TOOL_DEFINITION as CODE_EXECUTOR_DEF
from src.tools.builtin.code_validator import TOOL_DEFINITION as CODE_VALIDATOR_DEF
from src.tools.builtin.corpus import TOOL_DEFINITION_INDEX as CORPUS_INDEX_DEF
from src.tools.builtin.corpus import TOOL_DEFINITION_SEARCH as CORPUS_SEARCH_DEF
from src.tools.builtin.document_parser import TOOL_DEFINITION as DOCUMENT_PARSER_DEF
from src.tools.builtin.environment_inspect import TOOL_DEFINITION as ENVIRONMENT_INSPECT_DEF
from src.tools.builtin.file_reader import TOOL_DEFINITION as FILE_READER_DEF
from src.tools.builtin.file_writer import TOOL_DEFINITION as FILE_WRITER_DEF
from src.tools.builtin.get_current_time import TOOL_DEFINITION as GET_CURRENT_TIME_DEF
from src.tools.builtin.git_clone import TOOL_DEFINITION_CLONE as GIT_CLONE_DEF
from src.tools.builtin.git_clone import TOOL_DEFINITION_SEARCH as CODE_SEARCH_DEF
from src.tools.builtin.http_request import TOOL_DEFINITION as HTTP_REQUEST_DEF
from src.tools.builtin.image_generator import TOOL_DEFINITION as IMAGE_GENERATOR_DEF
from src.tools.builtin.list_directory import TOOL_DEFINITION as LIST_DIRECTORY_DEF
from src.tools.builtin.memory_search import TOOL_DEFINITION as MEMORY_SEARCH_DEF
from src.tools.builtin.ocr_parser import TOOL_DEFINITION as OCR_PARSER_DEF
from src.tools.builtin.self_inspect import TOOL_DEFINITION as SELF_INSPECT_DEF
from src.tools.builtin.schedule_task import TOOL_DEFINITION as SCHEDULE_TASK_DEF
from src.tools.builtin.terminal_command import TOOL_DEFINITION as TERMINAL_COMMAND_DEF
from src.tools.builtin.web_scraper import TOOL_DEFINITION as WEB_SCRAPER_DEF
from src.tools.builtin.web_search import TOOL_DEFINITION as WEB_SEARCH_DEF

ALL_TOOL_DEFINITIONS = [
    ARXIV_SEARCH_DEF,
    CODE_EXECUTOR_DEF,
    CODE_VALIDATOR_DEF,
    CODE_SEARCH_DEF,
    CORPUS_INDEX_DEF,
    CORPUS_SEARCH_DEF,
    DOCUMENT_PARSER_DEF,
    ENVIRONMENT_INSPECT_DEF,
    FILE_READER_DEF,
    FILE_WRITER_DEF,
    GET_CURRENT_TIME_DEF,
    GIT_CLONE_DEF,
    HTTP_REQUEST_DEF,
    IMAGE_GENERATOR_DEF,
    LIST_DIRECTORY_DEF,
    MEMORY_SEARCH_DEF,
    OCR_PARSER_DEF,
    SELF_INSPECT_DEF,
    SCHEDULE_TASK_DEF,
    TERMINAL_COMMAND_DEF,
    WEB_SCRAPER_DEF,
    WEB_SEARCH_DEF,
]

#: Coarse-grained capability tags (F3, findings-05) for scope-injection /
#: functional-dependency recall, plus MCP-style annotations. Keys are the four
#: MCP boolean hints — ``readOnlyHint`` / ``destructiveHint`` / ``idempotentHint``
#: / ``openWorldHint`` (omitted ⇒ False). The execute node routes any tool with
#: ``destructiveHint=True`` through a HITL gate (``DESTRUCTIVE_TOOL_HITL_ENABLED``,
#: default off). Annotations are applied by ``create_default_registry`` as a
#: fallback when a tool definition carries none of its own, so this stays a
#: single source of truth rather than 16 per-file edits.
#:
#: Destructive (gated): ``index_corpus`` (writes the corpus index), ``http_request``
#: (arbitrary network mutation), ``terminal_command`` (shell exec). ``file_writer``
#: and ``code_executor`` are intentionally NOT flagged destructive: the former is
#: path-confined + size-limited under ``RESULTS_PER_RUN_SUBDIR``, the latter runs
#: in the no-DinD runner sandbox — both already have their blast radius bounded,
#: so flagging them would gate safe compute behind a human prompt.
TOOL_ANNOTATIONS: dict[str, dict[str, object]] = {
    "arxiv_search": {"tags": ["search", "read"], "mcp_hints": {"readOnlyHint": True, "openWorldHint": True}},
    "code_executor": {"tags": ["compute"], "mcp_hints": {"openWorldHint": True}},
    "code_validator": {"tags": ["compute", "read"], "mcp_hints": {"readOnlyHint": True}},
    "code_search": {"tags": ["code", "search", "read"], "mcp_hints": {"readOnlyHint": True}},
    "corpus_search": {"tags": ["search", "read"], "mcp_hints": {"readOnlyHint": True}},
    "document_parser": {"tags": ["read", "compute"], "mcp_hints": {"readOnlyHint": True}},
    "environment_inspect": {"tags": ["read", "system"], "mcp_hints": {"readOnlyHint": True}},
    "file_reader": {"tags": ["read", "filesystem"], "mcp_hints": {"readOnlyHint": True}},
    "file_writer": {"tags": ["write", "filesystem"], "mcp_hints": {}},
    "get_current_time": {"tags": ["read", "system"], "mcp_hints": {"readOnlyHint": True, "idempotentHint": True}},
    "git_clone": {"tags": ["code", "fetch", "write"], "mcp_hints": {"openWorldHint": True}},
    "http_request": {"tags": ["network"], "mcp_hints": {"destructiveHint": True, "openWorldHint": True}},
    "image_generator": {"tags": ["generate", "write"], "mcp_hints": {"openWorldHint": True}},
    "index_corpus": {"tags": ["search", "write"], "mcp_hints": {"destructiveHint": True}},
    "list_directory": {"tags": ["read", "filesystem"], "mcp_hints": {"readOnlyHint": True}},
    "memory_search": {"tags": ["read", "memory"], "mcp_hints": {"readOnlyHint": True}},
    "ocr_parser": {"tags": ["read", "ocr"], "mcp_hints": {"readOnlyHint": True}},
    "self_inspect": {"tags": ["read", "system"], "mcp_hints": {"readOnlyHint": True}},
    "create_scheduled_task": {"tags": ["schedule", "write"], "mcp_hints": {"idempotentHint": True}},
    "terminal_command": {"tags": ["system", "write"], "mcp_hints": {"destructiveHint": True, "openWorldHint": True}},
    "web_scraper": {"tags": ["search", "read"], "mcp_hints": {"readOnlyHint": True, "openWorldHint": True}},
    "web_search": {"tags": ["search", "read"], "mcp_hints": {"readOnlyHint": True, "openWorldHint": True}},
}

__all__ = [
    "ALL_TOOL_DEFINITIONS",
    "TOOL_ANNOTATIONS",
]
