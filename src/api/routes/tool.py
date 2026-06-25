"""Operator tool-edit → review → approve HITL routes (D10).

Lets an operator edit a stored generated tool's code, validate it against the
SAME bar a runtime-generated tool must clear (D9's shared ``validate_tool_code``
— assert + ruff lint + 7-layer safety), stage the edit as a
``pending_review`` ToolVersion, and then approve/reject it. An approved edit
becomes the live version ``load_active_tools`` materializes on the next worker
start; a pending edit never runs until approved (its ``is_active=False`` +
``status='pending_review'`` keep it out of the registry).

The gate runs ``sandbox=None`` — the API process is stateless and (deliberately,
per the role-split) has no Docker access, so the optional functional sandbox
smoke (gate 4) is deferred to load/run. The deterministic gates 1-3 (assert +
lint + safety) are the correctness bar; a bad tool that slips the smoke fails
noisily at materialization/first-call, not silently in the API.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from src.safety.pipeline import SafetyPipeline
from src.tools.dynamic.persister import ToolPersister
from src.tools.dynamic.validation import validate_tool_code

router = APIRouter()

# Mount prefix for this router (set in app.py ``include_router``). Kept here so
# the route paths below stay self-describing.
API_PREFIX = "/api/v1/tools"

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class ToolEditRequest(BaseModel):
    """Body for ``PATCH /tools/{name}`` — an operator-authored tool edit.

    ``handler_code`` + ``test_code`` are required (D9: a tool must ship a test
    that asserts something). ``description``/``input_schema`` are optional and,
    when omitted, are carried over from the existing tool (a code-only edit
    keeps the registration's current description/schema).
    """

    handler_code: str = Field(..., min_length=1)
    test_code: str = Field(..., min_length=1)
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class ToolStagedResponse(BaseModel):
    """Acknowledgement that an edit was staged pending review (202 Accepted)."""

    tool_name: str
    version: int | None
    status: str
    detail: str


class ToolActionResponse(BaseModel):
    """Acknowledgement of an approve/reject action."""

    tool_name: str
    version: int | None
    status: str
    detail: str


class ToolListItem(BaseModel):
    """One row of ``GET /tools``."""

    tool_name: str
    description: str
    is_active: bool
    version: int | None = None
    status: str | None = None
    version_active: bool | None = None


class ToolListResponse(BaseModel):
    """Body of ``GET /tools``."""

    tools: list[ToolListItem]
    count: int


class ToolDetailResponse(BaseModel):
    """Body of ``GET /tools/{name}``."""

    tool_name: str
    description: str
    input_schema: dict[str, Any]
    is_active: bool
    version: int | None = None
    status: str | None = None
    code_content: str | None = None
    test_content: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


@router.patch(
    "/{name}",
    response_model=ToolStagedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def stage_tool_edit(name: str, body: ToolEditRequest) -> ToolStagedResponse:
    """Validate an operator edit and stage it as ``pending_review`` (D10).

    Runs the shared ``validate_tool_code`` gate (assert + ruff lint + 7-layer
    safety). On pass, the edit is persisted as a new ``pending_review``
    ToolVersion (not live); on fail, ``422`` with the first failing reason.
    """
    validation = await validate_tool_code(
        handler_code=body.handler_code,
        test_code=body.test_code,
        tool_name=name,
        safety_pipeline=SafetyPipeline(),
        sandbox=None,
        allowlisted_modules=None,
    )
    if not validation.passed:
        logger.info(f"Tool edit '{name}' rejected by code gate: {validation.reason}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=validation.reason,
        )

    persister = ToolPersister()

    # Carry over description/input_schema from the existing tool when the edit
    # omits them (a code-only edit keeps the registration's current metadata).
    existing = await persister.get_tool(name)
    effective_desc = body.description
    if effective_desc is None:
        if existing is None:
            # description is NOT NULL on the registration — a brand-new tool
            # requires one.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="description is required when creating a new tool",
            )
        effective_desc = existing["description"]
    effective_schema = body.input_schema or (
        (existing or {}).get("input_schema") or _EMPTY_SCHEMA
    )

    version_id = await persister.submit_pending_version(
        tool_name=name,
        description=effective_desc,
        input_schema=effective_schema,
        handler_code=body.handler_code,
        test_code=body.test_code,
    )
    if version_id is None:
        logger.warning(f"Tool edit '{name}' passed the gate but failed to stage")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tool edit validated but could not be staged (database unavailable)",
        )

    pending_version = None
    staged = await persister.get_tool(name)
    if staged is not None:
        pending_version = staged.get("version")

    return ToolStagedResponse(
        tool_name=name,
        version=pending_version,
        status="pending_review",
        detail="Edit validated and staged; awaiting approval (POST /approve).",
    )


@router.post(
    "/{name}/approve",
    response_model=ToolActionResponse,
)
async def approve_tool(name: str) -> ToolActionResponse:
    """Promote the latest ``pending_review`` version to live (D10)."""
    persister = ToolPersister()
    result = await persister.approve_pending(name)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending_review version found for tool '{name}'",
        )
    return ToolActionResponse(
        tool_name=result["tool_name"],
        version=result.get("version"),
        status=result["status"],
        detail="Version approved; it will load on the next worker start.",
    )


@router.post(
    "/{name}/reject",
    response_model=ToolActionResponse,
)
async def reject_tool(name: str, reason: str | None = None) -> ToolActionResponse:
    """Dismiss the latest ``pending_review`` version (D10)."""
    persister = ToolPersister()
    rejected = await persister.reject_pending(name, reason=reason)
    if not rejected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending_review version found for tool '{name}'",
        )
    # Read back the (now rejected) latest version number for the response.
    version: int | None = None
    detail_obj = await persister.get_tool(name)
    if detail_obj is not None:
        version = detail_obj.get("version")
    return ToolActionResponse(
        tool_name=name,
        version=version,
        status="rejected",
        detail="Pending version rejected; the live tool is unchanged.",
    )


@router.get("", response_model=ToolListResponse)
async def list_tools() -> ToolListResponse:
    """List generated tools + their latest version/status (D10)."""
    persister = ToolPersister()
    tools = await persister.list_tools()
    return ToolListResponse(
        tools=[ToolListItem(**t) for t in tools],
        count=len(tools),
    )


@router.get("/{name}", response_model=ToolDetailResponse)
async def get_tool(name: str) -> ToolDetailResponse:
    """Inspect a single generated tool + its version history (D10)."""
    persister = ToolPersister()
    detail = await persister.get_tool(name)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No tool '{name}' found",
        )
    return ToolDetailResponse(**detail)
