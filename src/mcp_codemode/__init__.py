"""
MCP Code Mode - Universal Python Code Execution Server

Works with ANY MCP client: Goose, Claude Desktop, Cursor, VS Code, etc.

Inspired by Cloudflare's Code Mode: LLMs are better at writing code
than making tool calls because they've trained on millions of real repos.
"""

import argparse
from .server import mcp

__version__ = "0.2.4"
__author__ = "Anas Eqal"


def main():
    """MCP Code Mode: Universal Python code execution server."""
    parser = argparse.ArgumentParser(
        description="Universal Python code execution MCP server - one tool to do anything"
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"mcp-pyrunner {__version__}"
    )
    parser.parse_args()
    mcp.run()


if __name__ == "__main__":
    main()
