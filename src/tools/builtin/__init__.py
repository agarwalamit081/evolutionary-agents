"""Built-in tools package — 14 tools for the agent."""

from src.tools.builtin.code_executor import TOOL_DEFINITION as CODE_EXECUTOR_DEF
from src.tools.builtin.code_validator import TOOL_DEFINITION as CODE_VALIDATOR_DEF
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
    CODE_EXECUTOR_DEF,
    CODE_VALIDATOR_DEF,
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
