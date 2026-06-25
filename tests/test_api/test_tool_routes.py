"""Tests for src.api.routes.tool — D10 operator edit→review→approve HITL API.

FastAPI TestClient tests. The code gate (``validate_tool_code``) and the
ToolPersister are mocked so these are fast/deterministic and never touch a DB or
spawn a sandbox. The gate's own correctness is covered by
tests/test_tools/test_dynamic/test_generator.py (TestValidateToolCode); here we
assert the HTTP contract: a passing edit is staged pending_review, a failing
gate is 422, approve/reject/list/get round-trip through the persister.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app

_PREFIX = "/api/v1/tools"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def mock_persister() -> Any:
    """Patch ToolPersister in the route module with a mock instance."""
    instance = MagicMock()
    instance.get_tool = AsyncMock(return_value=None)
    instance.submit_pending_version = AsyncMock(return_value=None)
    instance.approve_pending = AsyncMock(return_value=None)
    instance.reject_pending = AsyncMock(return_value=False)
    instance.list_tools = AsyncMock(return_value=[])
    with patch("src.api.routes.tool.ToolPersister", return_value=instance):
        yield instance


@pytest.fixture
def mock_gate() -> Any:
    """Patch validate_tool_code to a controllable async mock (passes by default)."""
    gate = AsyncMock()
    passed = MagicMock()
    passed.passed = True
    passed.reason = ""
    gate.return_value = passed
    with patch("src.api.routes.tool.validate_tool_code", new=gate):
        yield gate


# ---------------------------------------------------------------------------
# PATCH /tools/{name}
# ---------------------------------------------------------------------------


class TestPatchEdit:
    def test_valid_edit_stages_pending_review(
        self, client: TestClient, mock_gate: Any, mock_persister: Any
    ) -> None:
        # existing tool → description/schema carried over; staged version read back.
        mock_persister.get_tool.side_effect = [
            {"description": "old", "input_schema": {"type": "object"}},
            {"version": 3},
        ]
        mock_persister.submit_pending_version.return_value = uuid.uuid4()

        resp = client.patch(
            f"{_PREFIX}/my_tool",
            json={"handler_code": "async def h():\n    pass\n", "test_code": "assert True\n"},
        )

        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["tool_name"] == "my_tool"
        assert body["status"] == "pending_review"
        assert body["version"] == 3
        mock_persister.submit_pending_version.assert_awaited_once()
        # description was carried over from the existing tool.
        kwargs = mock_persister.submit_pending_version.await_args.kwargs
        assert kwargs["description"] == "old"

    def test_failing_gate_returns_422(
        self, client: TestClient, mock_gate: Any, mock_persister: Any
    ) -> None:
        failed = MagicMock()
        failed.passed = False
        failed.reason = "test_code must contain at least one assert statement"
        mock_gate.return_value = failed

        resp = client.patch(
            f"{_PREFIX}/my_tool",
            json={"handler_code": "x", "test_code": "no assert here"},
        )

        assert resp.status_code == 422
        assert "assert" in resp.json()["detail"]
        # A rejected edit never reaches the persister.
        mock_persister.submit_pending_version.assert_not_awaited()

    def test_new_tool_without_description_is_422(
        self, client: TestClient, mock_gate: Any, mock_persister: Any
    ) -> None:
        mock_persister.get_tool.return_value = None  # brand-new tool

        resp = client.patch(
            f"{_PREFIX}/new_tool",
            json={"handler_code": "async def h():\n    pass\n", "test_code": "assert True\n"},
        )

        assert resp.status_code == 422
        assert "description" in resp.json()["detail"]

    def test_submit_failure_returns_503(
        self, client: TestClient, mock_gate: Any, mock_persister: Any
    ) -> None:
        mock_persister.get_tool.return_value = {"description": "d", "input_schema": {}}
        mock_persister.submit_pending_version.return_value = None  # DB unavailable

        resp = client.patch(
            f"{_PREFIX}/my_tool",
            json={"handler_code": "async def h():\n    pass\n", "test_code": "assert True\n"},
        )

        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /tools/{name}/approve | /reject
# ---------------------------------------------------------------------------


class TestApproveReject:
    def test_approve_returns_200(self, client: TestClient, mock_persister: Any) -> None:
        mock_persister.approve_pending.return_value = {
            "tool_name": "my_tool",
            "version": 3,
            "status": "approved",
        }
        resp = client.post(f"{_PREFIX}/my_tool/approve")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "approved"
        assert body["version"] == 3

    def test_approve_no_pending_returns_404(
        self, client: TestClient, mock_persister: Any
    ) -> None:
        mock_persister.approve_pending.return_value = None
        resp = client.post(f"{_PREFIX}/my_tool/approve")
        assert resp.status_code == 404

    def test_reject_returns_200(self, client: TestClient, mock_persister: Any) -> None:
        mock_persister.reject_pending.return_value = True
        mock_persister.get_tool.return_value = {"version": 3}
        resp = client.post(f"{_PREFIX}/my_tool/reject")
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_reject_no_pending_returns_404(
        self, client: TestClient, mock_persister: Any
    ) -> None:
        mock_persister.reject_pending.return_value = False
        resp = client.post(f"{_PREFIX}/my_tool/reject")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /tools | /tools/{name}
# ---------------------------------------------------------------------------


class TestInspect:
    def test_list_tools(self, client: TestClient, mock_persister: Any) -> None:
        mock_persister.list_tools.return_value = [
            {
                "tool_name": "a",
                "description": "d",
                "is_active": True,
                "version": 1,
                "status": "approved",
                "version_active": True,
            }
        ]
        resp = client.get(_PREFIX)
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["tools"][0]["tool_name"] == "a"

    def test_get_tool(self, client: TestClient, mock_persister: Any) -> None:
        mock_persister.get_tool.return_value = {
            "tool_name": "a",
            "description": "d",
            "input_schema": {},
            "is_active": True,
            "version": 1,
            "status": "approved",
            "code_content": "x",
            "test_content": "y",
            "history": [],
        }
        resp = client.get(f"{_PREFIX}/a")
        assert resp.status_code == 200
        assert resp.json()["tool_name"] == "a"

    def test_get_missing_returns_404(
        self, client: TestClient, mock_persister: Any
    ) -> None:
        mock_persister.get_tool.return_value = None
        resp = client.get(f"{_PREFIX}/missing")
        assert resp.status_code == 404
