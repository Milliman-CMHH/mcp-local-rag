"""Application telemetry configuration.

Configures logging to stderr (visible in VS Code MCP server output) and
optionally exports telemetry to Azure Monitor (Application Insights) when
APPLICATIONINSIGHTS_CONNECTION_STRING is set. Requires the ``telemetry``
extra (``uv add mcp-local-rag[telemetry]``). When not set or when the
extra is not installed, telemetry is a no-op so local use works without
overhead.
"""

import logging
import os
import sys

LOGGER_NAME = "mcp_local_rag"

logger = logging.getLogger(LOGGER_NAME)


def configure_telemetry() -> None:
    """Bootstrap logging and optional Azure Monitor telemetry."""
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logger.addHandler(handler)

    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        logger.info(
            "APPLICATIONINSIGHTS_CONNECTION_STRING not set — telemetry disabled"
        )
        return

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor  # pyright: ignore[reportUnknownVariableType]
    except ImportError:
        logger.warning(
            "azure-monitor-opentelemetry is not installed — "
            "install with: uv add mcp-local-rag[telemetry]"
        )
        return

    configure_azure_monitor(  # pyright: ignore[reportUnknownMemberType]
        connection_string=connection_string,
        enable_live_metrics=True,
        logger_name=LOGGER_NAME,
    )
    logger.info("Azure Monitor telemetry configured")
