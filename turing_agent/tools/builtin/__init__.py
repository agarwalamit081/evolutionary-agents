"""Built-in tools package — 7 tools for the agent."""

from turing_agent.tools.builtin.code_executor import TOOL_DEFINITION as CODE_EXECUTOR_DEF
from turing_agent.tools.builtin.code_validator import TOOL_DEFINITION as CODE_VALIDATOR_DEF
from turing_agent.tools.builtin.file_reader import TOOL_DEFINITION as FILE_READER_DEF
from turing_agent.tools.builtin.file_writer import TOOL_DEFINITION as FILE_WRITER_DEF
from turing_agent.tools.builtin.memory_search import TOOL_DEFINITION as MEMORY_SEARCH_DEF
from turing_agent.tools.builtin.self_inspect import TOOL_DEFINITION as SELF_INSPECT_DEF
from turing_agent.tools.builtin.web_search import TOOL_DEFINITION as WEB_SEARCH_DEF

ALL_TOOL_DEFINITIONS = [
    CODE_EXECUTOR_DEF,
    CODE_VALIDATOR_DEF,
    FILE_READER_DEF,
    FILE_WRITER_DEF,
    MEMORY_SEARCH_DEF,
    SELF_INSPECT_DEF,
    WEB_SEARCH_DEF,
]

__all__ = [
    "ALL_TOOL_DEFINITIONS",
]
