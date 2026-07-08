"""Application telemetry configuration.

Configures logging to stderr (visible in VS Code MCP server output) and
optionally exports telemetry to Azure Monitor (Application Insights) when
APPLICATIONINSIGHTS_CONNECTION_STRING is set. Requires the ``azure`` extra.
When not set or when the extra is not installed, telemetry is a no-op so
local use works without overhead.

Call order
----------
1. ``configure_logging()``  — cheap, must happen before ``mcp.run()`` so log
   output is visible from the very first line.
2. ``configure_azure_monitor_async()``  — makes an outbound HTTPS call to
   register with Azure Monitor; run this in the background init task so it
   never delays the MCP server handshake.
"""

import logging
import os
import sys

LOGGER_NAME = "mcp_local_rag"

logger = logging.getLogger(LOGGER_NAME)


def configure_logging() -> None:
    """Set up stderr logging only — no network calls."""
    logger.setLevel(logging.INFO)
    # Guard against duplicate handlers if called more than once (e.g. in tests).
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        for h in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        logger.addHandler(handler)


async def configure_azure_monitor_async() -> None:
    """Register with Azure Monitor in a thread so it doesn't block the event loop.

    No-op when APPLICATIONINSIGHTS_CONNECTION_STRING is not set or when the
    azure-monitor-opentelemetry package is not installed.
    """
    import asyncio  # noqa: PLC0415

    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.info(
            "APPLICATIONINSIGHTS_CONNECTION_STRING not set — telemetry disabled"
        )
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor  # pyright: ignore[reportUnknownVariableType]  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "azure-monitor-opentelemetry is not installed — "
            "install with: uv add mcp-local-rag[azure]"
        )
        return

    await asyncio.to_thread(
        configure_azure_monitor,  # pyright: ignore[reportUnknownArgumentType]
        connection_string=connection_string,
        enable_live_metrics=True,
        logger_name=LOGGER_NAME,
    )
    logger.info("Azure Monitor telemetry configured")


# Keep the old name as a convenience shim for any callers that still use it.
def configure_telemetry() -> None:
    """Deprecated shim: sets up logging only. Azure Monitor must be configured
    separately via ``configure_azure_monitor_async()``."""
    configure_logging()
