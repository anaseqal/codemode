# MCP Code Mode 🐍⚡

**Universal Python code execution MCP server - one tool to rule them all.**

Inspired by [Cloudflare's Code Mode](https://blog.cloudflare.com/code-mode/): LLMs are better at writing code than making tool calls because they've trained on millions of real repositories.

## Why Code Mode?

**Traditional approach** (many tools):
```
User: "Get weather for Austin and save to file"

LLM: [tool_call: get_weather(location="Austin")]
     → waits for response...
LLM: [tool_call: write_file(path="weather.txt", content=...)]
     → waits for response...
```

**Code Mode approach** (one tool):
```
User: "Get weather for Austin and save to file"

LLM: [run_python]
import requests
weather = requests.get("https://wttr.in/Austin?format=j1").json()
temp = weather['current_condition'][0]['temp_F']
with open("weather.txt", "w") as f:
    f.write(f"Austin: {temp}°F")
print(f"Saved! Temperature: {temp}°F")
```

### Benefits

| Traditional Tools | Code Mode |
|------------------|-----------|
| ❌ LLMs struggle with synthetic tool-call format | ✅ LLMs excel at writing real code |
| ❌ Each tool call = round trip to LLM | ✅ Complex workflows in one execution |
| ❌ Managing 20+ extensions | ✅ One universal tool |
| ❌ Token waste passing data between calls | ✅ Efficient data flow in code |
| ❌ Limited to pre-built capabilities | ✅ Anything Python can do |

## Works With Any MCP Client

- ✅ **Goose** (Block's AI agent)
- ✅ **Claude Desktop**
- ✅ **Cursor**
- ✅ **VS Code with Copilot**
- ✅ **Any MCP-compatible agent**

## Features

### 🚀 Universal Execution
Write Python to accomplish any task - HTTP requests, file operations, data processing, web scraping, image manipulation, and more.

### 📦 Auto-Install Dependencies
Missing a package? Code Mode detects `ModuleNotFoundError`, installs the package, and retries automatically.

### 🌊 Streaming Output
See results in real-time! `run_python_stream` shows output line-by-line as your code executes. Perfect for long-running tasks, progress bars, and monitoring live operations.

### 🖼️ Automatic File Display (Goose Compatible!)
Generated images, logs, or data files? Code Mode automatically detects and displays them in your MCP client! Supports:
- **Images**: PNG, JPG, GIF, SVG, HEIC, TIFF, etc. (displayed inline)
- **Text Files**: JSON, logs, source code (Python, JS, TS, Go, Rust, etc.), CSV, YAML, etc. (shown with syntax highlighting)
- **Resources**: PDFs, archives, videos (MP4, MOV), audio (MP3, WAV), Office docs, databases (available for download)

Just print the file path and Code Mode handles the rest! Works seamlessly with Goose and other MCP clients.

### 🧠 Dual Learning System (Enhanced!)
Records **both error-based and semantic failures**:
- **Error Learning**: Captures errors (ModuleNotFoundError, SSL errors, etc.) and their solutions
- **Semantic Learning**: Learns when code runs successfully but doesn't accomplish the objective

Future executions benefit from past learnings. Persists across sessions.

### 🔄 Intelligent Retry
`run_with_retry` analyzes failures and suggests fixes based on both error patterns and semantic learnings from similar tasks.

### 🐳 Optional Docker Sandbox
Run code in isolated Docker containers for enhanced security.

### ⚙️ Configurable
Adjust timeouts, execution modes, package restrictions, and more.

## Installation

### From PyPI (when published)

```bash
# Using uv (recommended)
uv tool install mcp-pyrunner

# Using pip
pip install mcp-pyrunner
```

### From Source

```bash
git clone https://github.com/anaseqal/codemode.git
cd codemode
uv sync
```

## Configuration

### Goose

**If installed from PyPI:**

Edit `~/.config/goose/config.yaml`:

```yaml
extensions:
  codemode:
    type: stdio
    enabled: true
    cmd: uvx
    args: ["mcp-pyrunner"]
```

**If running from source (local development):**

```yaml
extensions:
  codemode:
    type: stdio
    enabled: true
    cmd: uv
    args: ["run", "--directory", "/path/to/codemode", "mcp-pyrunner"]
    # Replace /path/to/codemode with actual path (e.g., ~/codemode)
```

Or use the UI: Extensions → Add Custom Extension → STDIO → Command: `uv run --directory /path/to/codemode mcp-pyrunner`

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

**If installed from PyPI:**
```json
{
  "mcpServers": {
    "codemode": {
      "command": "uvx",
      "args": ["mcp-pyrunner"]
    }
  }
}
```

**If running from source:**
```json
{
  "mcpServers": {
    "codemode": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/codemode", "mcp-pyrunner"]
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json`:

**If installed from PyPI:**
```json
{
  "mcpServers": {
    "codemode": {
      "command": "uvx",
      "args": ["mcp-pyrunner"]
    }
  }
}
```

**If running from source:**
```json
{
  "mcpServers": {
    "codemode": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/codemode", "mcp-pyrunner"]
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_system_context` | Get environment info (OS, Python, pip versions, package managers, learnings) |
| `run_python` | Execute Python code (auto-installs packages, auto-displays files) |
| `run_python_stream` | Execute with **real-time streaming output** (auto-displays files) |
| `run_with_retry` | Execute with intelligent retry, error analysis, and semantic learning suggestions |
| `add_learning` | Record error-based solutions for future reference |
| `record_semantic_failure` | **NEW!** Record when code runs but doesn't accomplish objective |
| `get_learnings` | View/search past learnings (both error and semantic) |
| `pip_install` | Pre-install a specific package |
| `configure` | View/update settings |

## Usage Examples

### Web Scraping

```
User: "Scrape the top 10 posts from Hacker News"

→ run_python:
import requests
from bs4 import BeautifulSoup

resp = requests.get("https://news.ycombinator.com")
soup = BeautifulSoup(resp.text, "html.parser")

for i, item in enumerate(soup.select(".titleline > a")[:10], 1):
    print(f"{i}. {item.text}")
    print(f"   {item['href']}\n")
```

### Data Processing

```
User: "Analyze sales.csv and show monthly totals"

→ run_python:
import pandas as pd

df = pd.read_csv("sales.csv")
df["date"] = pd.to_datetime(df["date"])
monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()

print("Monthly Sales:")
for period, total in monthly.items():
    print(f"  {period}: ${total:,.2f}")
```

### API Integration

```
User: "Get the current Bitcoin price in USD"

→ run_python:
import requests

data = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot").json()
price = float(data["data"]["amount"])
print(f"Bitcoin: ${price:,.2f} USD")
```

### Image Processing

```
User: "Resize all images in ./photos to 800x600"

→ run_python:
from pathlib import Path
from PIL import Image

photos = Path("./photos")
for img_path in photos.glob("*.jpg"):
    img = Image.open(img_path)
    img.thumbnail((800, 600))
    img.save(img_path)
    print(f"Resized: {img_path.name}")
```

### Streaming Output (Real-Time Progress)

```
User: "Scrape top 20 HN posts with progress updates"

→ run_python_stream:
import requests
from bs4 import BeautifulSoup
import time

print("🔍 Starting to scrape Hacker News...")

resp = requests.get("https://news.ycombinator.com")
soup = BeautifulSoup(resp.text, "html.parser")
stories = soup.select(".titleline > a")[:20]

print(f"📊 Found {len(stories)} stories. Processing...\n")

for i, story in enumerate(stories, 1):
    # Show progress in real-time
    progress = "█" * i + "░" * (20 - i)
    print(f"[{progress}] {i}/20: {story.text}")
    time.sleep(0.5)  # See each item appear live!

print("\n✅ Scraping complete!")

# Output appears LINE BY LINE as the code runs,
# not all at once at the end!
```

### Automatic File Display

```
User: "Take a screenshot of example.com and create a summary report"

→ run_python:
from playwright.sync_api import sync_playwright
import json

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://example.com")

    # Take screenshot
    screenshot_path = "/tmp/example_screenshot.png"
    page.screenshot(path=screenshot_path)

    # Create report
    report = {
        "url": "https://example.com",
        "title": page.title(),
        "screenshot": screenshot_path,
        "timestamp": "2025-01-26T12:00:00"
    }

    report_path = "/tmp/report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    browser.close()

    # Print file paths - Code Mode auto-detects and displays them!
    print(f"Screenshot saved to: {screenshot_path}")
    print(f"Report saved to: {report_path}")

# Result: Your MCP client displays the screenshot IMAGE inline
# and shows the JSON content formatted - no manual handling needed!
```

## Configuration Options

View current config:
```
→ configure()
```

Update settings:
```
→ configure(action="set", key="execution_mode", value="docker")
→ configure(action="set", key="default_timeout", value="120")
```

| Setting | Values | Description |
|---------|--------|-------------|
| `execution_mode` | `direct`, `docker` | How to run code |
| `default_timeout` | integer | Default timeout (seconds) |
| `max_retries` | integer | Default retry attempts |
| `auto_install` | `true`, `false` | Auto-install packages |
| `docker_image` | string | Docker image for sandbox |

## Dual Learning System

Code Mode learns from **two types of failures**:

### 1. Error-Based Learning
When you solve an error, record it:

```
→ add_learning(
    error_pattern="SSL: CERTIFICATE_VERIFY_FAILED",
    solution="Use verify=False or install/update certifi",
    context="HTTPS requests on systems with cert issues",
    tags="ssl,https,certificates"
)
```

### 2. Semantic Learning (NEW!)
When code runs successfully but doesn't accomplish the objective:

```
→ record_semantic_failure(
    objective="Display image in Goose app",
    failed_approach="Used print() to output file path",
    successful_approach="Returned base64 encoded image as MCP content object",
    context="MCP clients need structured content objects, not just paths",
    tags="goose,mcp,display,images"
)
```

**Why semantic learning matters:**
- Code executed without errors ≠ objective accomplished
- AI learns from "technically correct but semantically wrong" approaches
- Future attempts at similar objectives benefit from past semantic learnings

### View Learnings
```
→ get_learnings()              # View all learnings (error + semantic)
→ get_learnings(search="ssl")  # Search learnings
```

**Learnings are distinguished by icons:**
- 🔴 Error-based learnings
- 🔵 Semantic learnings

Learnings persist in `~/.mcp-pyrunner/learnings.json` and improve future executions.

## Data Storage

Code Mode stores data in `~/.mcp-pyrunner/`:

```
~/.mcp-pyrunner/
├── config.json       # User configuration
├── learnings.json    # Error patterns and solutions
└── execution_log.json # Recent execution history
```

## Security Considerations

⚠️ **Code Mode executes arbitrary Python code.**

**Direct mode** (default):
- Code runs with your user permissions
- Full filesystem and network access
- Fast execution

**Docker mode** (more secure):
- Code runs in isolated container
- Limited resources (512MB RAM, 1 CPU)
- Network access available
- Slower startup

Enable Docker mode:
```
→ configure(action="set", key="execution_mode", value="docker")
```

## Testing

```bash
# Run tests
uv run pytest

# Test with MCP Inspector
uv run mcp dev src/mcp_codemode/server.py
# Open http://localhost:5173
```

## Contributing

Contributions welcome! Areas of interest:

- [x] ~~Streaming output for long-running code~~ ✅ **DONE!**
- [x] ~~Automatic file display (images, text, resources)~~ ✅ **DONE!**
- [x] ~~Enhanced system context (pip version, package managers)~~ ✅ **DONE!**
- [ ] Vector DB for semantic learning search
- [ ] Pyodide/WASM sandboxing option
- [ ] Code analysis before execution
- [ ] Resource usage tracking
- [ ] Multi-file project support

## License

MIT

## Acknowledgments

- [Cloudflare's Code Mode](https://blog.cloudflare.com/code-mode/) for the inspiration
- [Model Context Protocol](https://modelcontextprotocol.io/) for the standard
- [Block's Goose](https://github.com/block/goose) for being an excellent MCP client
