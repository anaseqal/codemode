#!/bin/bash
# MCP Code Mode - Setup and Test Script
# Run this after downloading the project

set -e

echo "🐍 MCP Code Mode Setup"
echo "======================"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "📦 Installing dependencies..."
uv sync

#echo ""
#echo "🧪 Running tests..."
#uv run pytest tests/ -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "To add to Goose, edit ~/.config/goose/config.yaml:"
echo ""
echo "extensions:"
echo "  codemode:"
echo "    type: stdio"
echo "    enabled: true"
echo "    cmd: uv"
echo "    args:"
echo "      - run"
echo "      - $(pwd)/.venv/bin/mcp-codemode"
echo ""
echo "Or use the full path:"
echo "    cmd: $(pwd)/.venv/bin/mcp-codemode"
echo ""
echo "For Claude Desktop, edit ~/Library/Application Support/Claude/claude_desktop_config.json:"
echo '{'
echo '  "mcpServers": {'
echo '    "codemode": {'
echo '      "command": "'$(pwd)'/.venv/bin/mcp-codemode"'
echo '    }'
echo '  }'
echo '}'
echo ""
echo "🎉 Ready to use! Start Goose and try: 'Write Python code to list files in current directory'"
