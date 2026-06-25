"""Built-in tools package — 17 distinct tools for the agent.

The original 14 tools were audited for true duplicates (M7b); the 2 corpus
tools (Phase 1 search stack) are additive — `index_corpus` (write) and
`corpus_search` (read) are a distinct gather/recall cluster, not a dup of
`web_search` (live) or `web_scraper` (single page). All names and descriptions
remain unique; the similar clusters (listing / reading / fetching) are
deliberately distinct, not mergeable. ``test_consolidation.py`` locks this in
— it fails if a future change duplicates a name/description or collapses a
cluster, preventing the B3 capability bloat that dynamic-tool dedup addresses.
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
from src.tools.builtin.http_request import TOOL_DEFINITION as HTTP_REQUEST_DEF
from src.tools.builtin.list_directory import TOOL_DEFINITION as LIST_DIRECTORY_DEF
from src.tools.builtin.memory_search import TOOL_DEFINITION as MEMORY_SEARCH_DEF
from src.tools.builtin.self_inspect import TOOL_DEFINITION as SELF_INSPECT_DEF
from src.tools.builtin.terminal_command import TOOL_DEFINITION as TERMINAL_COMMAND_DEF
from src.tools.builtin.web_scraper import TOOL_DEFINITION as WEB_SCRAPER_DEF
from src.tools.builtin.web_search import TOOL_DEFINITION as WEB_SEARCH_DEF

ALL_TOOL_DEFINITIONS = [
    ARXIV_SEARCH_DEF,
    CODE_EXECUTOR_DEF,
    CODE_VALIDATOR_DEF,
    CORPUS_INDEX_DEF,
    CORPUS_SEARCH_DEF,
    DOCUMENT_PARSER_DEF,
    ENVIRONMENT_INSPECT_DEF,
    FILE_READER_DEF,
    FILE_WRITER_DEF,
    GET_CURRENT_TIME_DEF,
    HTTP_REQUEST_DEF,
    LIST_DIRECTORY_DEF,
    MEMORY_SEARCH_DEF,
    SELF_INSPECT_DEF,
    TERMINAL_COMMAND_DEF,
    WEB_SCRAPER_DEF,
    WEB_SEARCH_DEF,
]

__all__ = [
    "ALL_TOOL_DEFINITIONS",
]
