"""DB persistence for dynamically generated tools.

Stores validated tools in ToolRegistration + ToolVersion tables
and loads them back into ToolRegistry at startup.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger


if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class ToolPersister:
    """Persist and load dynamically generated tools from the database."""

    async def persist(
        self,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        handler_code: str,
        test_code: str = "",
    ) -> uuid.UUID | None:
        """Write ToolRegistration + ToolVersion to DB.

        Args:
            tool_name: Unique snake_case identifier.
            description: Human-readable tool description.
            input_schema: JSON Schema for tool parameters.
            handler_code: Complete async function source code.
            test_code: Optional test code for the tool.

        Returns:
            UUID of the ToolRegistration row, or None on failure.
        """
        try:
            from src.db.models import ToolRegistration, ToolVersion
            from src.db.session import get_session

            async with get_session() as session:
                # Check if tool already exists (update instead of duplicate)
                from sqlalchemy import select

                stmt = select(ToolRegistration).where(
                    ToolRegistration.tool_name == tool_name
                )
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()

                if existing is not None:
                    # Create a new version of the existing tool
                    version_stmt = select(ToolVersion).where(
                        ToolVersion.tool_id == existing.id
                    ).order_by(ToolVersion.version.desc())
                    ver_result = await session.execute(version_stmt)
                    latest_ver = ver_result.scalars().first()
                    next_version = (latest_ver.version + 1) if latest_ver else 1

                    # Deactivate old versions
                    from sqlalchemy import update

                    await session.execute(
                        update(ToolVersion)
                        .where(ToolVersion.tool_id == existing.id)
                        .values(is_active=False)
                    )

                    new_version = ToolVersion(
                        tool_id=existing.id,
                        version=next_version,
                        code_content=handler_code,
                        test_content=test_code or None,
                        is_active=True,
                    )
                    session.add(new_version)
                    await session.flush()

                    logger.info(
                        f"Updated tool '{tool_name}' to version {next_version}"
                    )
                    return existing.id

                # Create new tool registration
                registration = ToolRegistration(
                    tool_name=tool_name,
                    tool_type="generated",
                    description=description,
                    input_schema=input_schema,
                    is_active=True,
                )
                session.add(registration)
                await session.flush()

                # Create initial version
                version = ToolVersion(
                    tool_id=registration.id,
                    version=1,
                    code_content=handler_code,
                    test_content=test_code or None,
                    is_active=True,
                )
                session.add(version)

                logger.info(f"Persisted new tool '{tool_name}' (version 1)")
                return registration.id

        except Exception as e:
            logger.warning(f"Failed to persist tool '{tool_name}': {e}")
            return None

    async def load_active_tools(
        self,
        registry: ToolRegistry,
    ) -> list[str]:
        """Load all active generated tools from DB and register them.

        Queries ToolRegistration where is_active=True, fetches the
        active ToolVersion, materializes the handler, and registers
        each tool in the provided ToolRegistry.

        Args:
            registry: ToolRegistry to register loaded tools into.

        Returns:
            List of loaded tool names.
        """
        loaded: list[str] = []

        try:
            from src.db.models import ToolRegistration, ToolVersion
            from src.db.session import get_session
            from sqlalchemy import select

            async with get_session() as session:
                # Find all active generated tools
                stmt = select(ToolRegistration).where(
                    ToolRegistration.is_active.is_(True),
                    ToolRegistration.tool_type == "generated",
                )
                result = await session.execute(stmt)
                registrations = result.scalars().all()

                for reg in registrations:
                    try:
                        # Get the active version
                        ver_stmt = select(ToolVersion).where(
                            ToolVersion.tool_id == reg.id,
                            ToolVersion.is_active.is_(True),
                        ).order_by(ToolVersion.version.desc()).limit(1)
                        ver_result = await session.execute(ver_stmt)
                        version = ver_result.scalar_one_or_none()

                        if version is None:
                            logger.debug(
                                f"No active version for tool '{reg.tool_name}', skipping"
                            )
                            continue

                        # Materialize and register
                        from src.tools.dynamic.generator import ToolGenerator

                        materializer = ToolGenerator.__new__(ToolGenerator)
                        handler = materializer._materialize_handler(
                            version.code_content
                        )

                        registry.register(
                            name=reg.tool_name,
                            handler=handler,
                            description=reg.description,
                            parameters=reg.input_schema,
                        )
                        loaded.append(reg.tool_name)

                    except Exception as e:
                        logger.warning(
                            f"Failed to load tool '{reg.tool_name}': {e}"
                        )

        except Exception as e:
            logger.debug(f"Could not load dynamic tools from DB: {e}")

        return loaded
