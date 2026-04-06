"""Connector interface for ingesting data from external systems into Attest."""

from __future__ import annotations

import importlib

from attestdb.connectors.base import (
    Connector,
    ConnectorResult,
    HybridConnector,
    QueryConnector,
    StructuredConnector,
    TextConnector,
)

# Lazy registry: name -> (module_path, class_name)
# Open-source connectors only. Enterprise connectors (Salesforce, ServiceNow,
# Jira, Confluence, Teams, Zendesk, SharePoint, Box, Zoho, DSI) are available
# in attestdb-enterprise.
CONNECTOR_REGISTRY: dict[str, tuple[str, str]] = {
    # Chat
    "slack": ("attestdb.connectors.slack", "SlackConnector"),
    # Email
    "gmail": ("attestdb.connectors.gmail", "GmailConnector"),
    # Docs
    "gdocs": ("attestdb.connectors.gdocs", "GDocsConnector"),
    "notion": ("attestdb.connectors.notion", "NotionConnector"),
    # Databases
    "postgres": ("attestdb.connectors.postgres", "PostgresConnector"),
    "mysql": ("attestdb.connectors.mysql", "MySQLConnector"),
    "mssql": ("attestdb.connectors.mssql", "MSSQLConnector"),
    "sqlite": ("attestdb.connectors.sqlite", "SQLiteConnector"),
    "mongodb": ("attestdb.connectors.mongodb", "MongoDBConnector"),
    # Search / Storage
    "elasticsearch": ("attestdb.connectors.elasticsearch", "ElasticsearchConnector"),
    "s3": ("attestdb.connectors.s3", "S3Connector"),
    "csv": ("attestdb.connectors.csv_connector", "CSVConnector"),
    "http": ("attestdb.connectors.http_connector", "HTTPConnector"),
    # Dev tools
    "github": ("attestdb.connectors.github", "GitHubConnector"),
    "linear": ("attestdb.connectors.linear", "LinearConnector"),
    "pagerduty": ("attestdb.connectors.pagerduty", "PagerDutyConnector"),
    # Productivity
    "google_sheets": ("attestdb.connectors.google_sheets", "GoogleSheetsConnector"),
    "gdrive": ("attestdb.connectors.gdrive", "GDriveConnector"),
    "airtable": ("attestdb.connectors.airtable", "AirtableConnector"),
    # CRM
    "hubspot": ("attestdb.connectors.hubspot", "HubSpotConnector"),
}


def connect(name: str, **kwargs) -> Connector:
    """Instantiate a connector by name.

    Args:
        name: Connector name (e.g. "slack", "postgres", "notion").
        **kwargs: Passed to the connector constructor.

    Returns:
        A :class:`Connector` instance ready for ``.run(db)``.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in CONNECTOR_REGISTRY:
        available = ", ".join(sorted(CONNECTOR_REGISTRY))
        raise KeyError(f"Unknown connector {name!r}. Available: {available}")
    module_path, class_name = CONNECTOR_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = [
    "Connector",
    "ConnectorResult",
    "CONNECTOR_REGISTRY",
    "HybridConnector",
    "QueryConnector",
    "StructuredConnector",
    "TextConnector",
    "connect",
]
