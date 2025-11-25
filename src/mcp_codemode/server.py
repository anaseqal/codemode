"""
MCP Code Mode - Universal Python Code Execution Server

Inspired by Cloudflare's "Code Mode": LLMs are better at writing code
than making tool calls because they've trained on millions of real repos.

Works with ANY MCP client:
- Goose
- Claude Desktop
- Cursor
- VS Code with Copilot
- Any MCP-compatible agent

Features:
- Execute any Python code to accomplish tasks
- Auto-install missing dependencies  
- Intelligent retry with learning from failures
- Persistent learning database
- Rich system context for better code generation
- Optional Docker sandboxing for security
"""

import subprocess
import tempfile
import os
import sys
import json
import platform
import hashlib
import traceback
import re
import shutil
import select
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Generator
from dataclasses import dataclass, asdict, field
from enum import Enum

from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR

# ============================================================================
# Configuration
# ============================================================================

class ExecutionMode(str, Enum):
    """Code execution modes"""
    DIRECT = "direct"  # Run directly (fast, less secure)
    DOCKER = "docker"  # Run in Docker container (slower, more secure)
    # Future: PYODIDE = "pyodide"  # Run in WASM sandbox

# Directory for persistent data
DATA_DIR = Path.home() / ".mcp-pyrunner"
LEARNING_FILE = DATA_DIR / "learnings.json"
EXECUTION_LOG = DATA_DIR / "execution_log.json"
CONFIG_FILE = DATA_DIR / "config.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    "execution_mode": ExecutionMode.DIRECT.value,
    "default_timeout": 60,
    "max_retries": 3,
    "auto_install": True,
    "docker_image": "python:3.12-slim",
    "allowed_packages": [],  # Empty = allow all
    "blocked_packages": ["os-sys"],  # Packages to never install
}

def load_config() -> dict:
    """Load or create configuration"""
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(config: dict):
    """Save configuration"""
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

CONFIG = load_config()

# Pre-installed libraries available for code execution
AVAILABLE_LIBRARIES = {
    # HTTP & Web
    "requests": "HTTP requests made easy",
    "httpx": "Modern async HTTP client",
    "aiohttp": "Async HTTP client/server",
    "beautifulsoup4": "HTML/XML parsing",
    "lxml": "Fast XML/HTML processing",
    "selenium": "Browser automation (if installed)",
    
    # Data Processing
    "pandas": "Data manipulation and analysis",
    "numpy": "Numerical computing",
    "json": "JSON encoding/decoding (stdlib)",
    "csv": "CSV file handling (stdlib)",
    "yaml": "YAML parsing (pyyaml)",
    
    # Files & System
    "pathlib": "Object-oriented paths (stdlib)",
    "os": "OS interface (stdlib)",
    "shutil": "File operations (stdlib)",
    "subprocess": "Process execution (stdlib)",
    "tempfile": "Temporary files (stdlib)",
    
    # Images
    "PIL": "Image processing (Pillow)",
    "cv2": "Computer vision (opencv-python)",
    
    # Utilities
    "re": "Regular expressions (stdlib)",
    "datetime": "Date/time handling (stdlib)",
    "hashlib": "Hashing (stdlib)",
    "base64": "Base64 encoding (stdlib)",
    "typing": "Type hints (stdlib)",
}

# Module name to pip package mapping
MODULE_TO_PACKAGE = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "skimage": "scikit-image",
}

# ============================================================================
# Initialize MCP Server
# ============================================================================

mcp = FastMCP(
    "mcp-codemode",
    instructions="""Universal Python code execution server.

Instead of using many specialized tools, write Python code to accomplish ANY task.
LLMs are better at writing code than making tool calls - leverage this!

Available tools:
- get_system_context: Get environment info before writing code
- run_python: Run Python code (auto-installs missing packages)
- run_python_stream: Run Python with real-time streaming output
- run_with_retry: Run with intelligent retry and error learning
- add_learning: Record solutions for future reference
- get_learnings: View past learnings
- pip_install: Pre-install a specific package
- configure: View/update settings

Tips:
- Always use print() to output results
- Use run_python_stream for long-running tasks to see progress in real-time
- Complex tasks can be done in one code block
- Packages are auto-installed on ImportError
- Learnings persist and improve future executions
"""
)

# ============================================================================
# Learning System
# ============================================================================

@dataclass
class Learning:
    """A learned pattern from execution errors"""
    error_pattern: str
    solution: str
    context: str
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)


class LearningStore:
    """Persistent store for execution learnings"""
    
    def __init__(self, filepath: Path = LEARNING_FILE):
        self.filepath = filepath
        self.learnings: list[Learning] = []
        self._load()
    
    def _load(self):
        if self.filepath.exists():
            try:
                data = json.loads(self.filepath.read_text())
                self.learnings = [Learning(**l) for l in data]
            except Exception:
                self.learnings = []
    
    def _save(self):
        data = [asdict(l) for l in self.learnings]
        self.filepath.write_text(json.dumps(data, indent=2))
    
    def add(self, error_pattern: str, solution: str, context: str, tags: list[str] = None):
        """Add or update a learning"""
        tags = tags or []
        
        # Check for existing pattern
        for l in self.learnings:
            if l.error_pattern == error_pattern:
                l.solution = solution
                l.context = context
                l.success_count += 1
                l.last_used = datetime.now().isoformat()
                l.tags = list(set(l.tags + tags))
                self._save()
                return
        
        self.learnings.append(Learning(
            error_pattern=error_pattern,
            solution=solution,
            context=context,
            tags=tags
        ))
        self._save()
    
    def find_relevant(self, error: str, code: str = "", limit: int = 5) -> list[Learning]:
        """Find learnings that might help with an error"""
        relevant = []
        error_lower = error.lower()
        
        for learning in self.learnings:
            score = 0
            try:
                if re.search(learning.error_pattern, error, re.IGNORECASE):
                    score += 10
            except re.error:
                if learning.error_pattern.lower() in error_lower:
                    score += 5
            
            # Boost by success rate
            if learning.success_count > 0:
                score += min(learning.success_count, 5)
            
            if score > 0:
                relevant.append((score, learning))
        
        relevant.sort(key=lambda x: -x[0])
        return [l for _, l in relevant[:limit]]
    
    def mark_success(self, error_pattern: str):
        """Mark a learning as successfully used"""
        for l in self.learnings:
            if l.error_pattern == error_pattern:
                l.success_count += 1
                l.last_used = datetime.now().isoformat()
                self._save()
                return
    
    def mark_failure(self, error_pattern: str):
        """Mark a learning as failed"""
        for l in self.learnings:
            if l.error_pattern == error_pattern:
                l.failure_count += 1
                self._save()
                return
    
    def get_summary(self) -> str:
        """Get a summary of all learnings"""
        if not self.learnings:
            return "No learnings recorded yet. Execute code and learn from errors!"
        
        lines = [f"📚 {len(self.learnings)} learnings recorded:\n"]
        
        # Sort by success count
        sorted_learnings = sorted(
            self.learnings, 
            key=lambda x: x.success_count, 
            reverse=True
        )
        
        for i, l in enumerate(sorted_learnings[:15], 1):
            pattern_short = l.error_pattern[:40] + "..." if len(l.error_pattern) > 40 else l.error_pattern
            lines.append(f"{i}. [{l.success_count}✓/{l.failure_count}✗] {pattern_short}")
            lines.append(f"   → {l.solution[:60]}...")
        
        if len(self.learnings) > 15:
            lines.append(f"\n   ... and {len(self.learnings) - 15} more")
        
        return "\n".join(lines)


# Global learning store
learning_store = LearningStore()

# ============================================================================
# System Information
# ============================================================================

def get_installed_packages() -> dict[str, str]:
    """Get installed pip packages"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            packages = json.loads(result.stdout)
            return {p["name"].lower(): p["version"] for p in packages}
    except Exception:
        pass
    return {}


def get_system_info() -> dict:
    """Gather comprehensive system information"""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "home": str(Path.home()),
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "execution_mode": CONFIG.get("execution_mode", "direct"),
        "docker_available": shutil.which("docker") is not None,
    }


def format_system_context() -> str:
    """Format system info as context for code generation"""
    info = get_system_info()
    packages = get_installed_packages()
    
    # Categorize packages
    installed_libs = []
    for mod, desc in AVAILABLE_LIBRARIES.items():
        pkg = MODULE_TO_PACKAGE.get(mod, mod).lower()
        if pkg in packages or mod.lower() in packages:
            installed_libs.append(f"  ✓ {mod}: {desc}")
        else:
            installed_libs.append(f"  ○ {mod}: {desc} (will auto-install)")
    
    context = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    MCP CODE MODE - SYSTEM CONTEXT                 ║
╚══════════════════════════════════════════════════════════════════╝

🖥️  SYSTEM
   OS: {info['os']} {info['os_release']}
   Python: {info['python_version']}
   Architecture: {info['architecture']}
   Execution Mode: {info['execution_mode']}
   Docker Available: {'Yes' if info['docker_available'] else 'No'}

📁 PATHS
   Working Directory: {info['cwd']}
   Home: {info['home']}
   Temp: {tempfile.gettempdir()}

📦 AVAILABLE LIBRARIES
{chr(10).join(installed_libs)}

⚙️  EXECUTION NOTES
   • Use print() to return results - stdout is captured
   • Missing packages are auto-installed on ImportError
   • Use pathlib.Path for cross-platform file operations
   • subprocess.run() available for shell commands
   • Network access available (requests, httpx, etc.)
   • Files in /tmp are safe for temporary storage

📚 LEARNINGS FROM PAST EXECUTIONS
{learning_store.get_summary()}

💡 TIPS
   • For complex tasks, break into logical steps
   • Handle errors with try/except for robustness
   • Use f-strings for clean output formatting
   • Prefer pathlib over os.path for file operations
""".strip()
    
    return context

# ============================================================================
# Code Execution Engine
# ============================================================================

def extract_missing_module(error: str) -> Optional[str]:
    """Extract missing module name from import error"""
    patterns = [
        r"No module named ['\"]([^'\"]+)['\"]",
        r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
        r"ImportError: cannot import name .+ from ['\"]([^'\"]+)['\"]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error)
        if match:
            return match.group(1).split(".")[0]
    return None


def install_package(package: str) -> tuple[bool, str]:
    """Install a pip package"""
    # Check blocklist
    if package.lower() in [p.lower() for p in CONFIG.get("blocked_packages", [])]:
        return False, f"Package '{package}' is blocked by configuration"
    
    # Check allowlist (if configured)
    allowed = CONFIG.get("allowed_packages", [])
    if allowed and package.lower() not in [p.lower() for p in allowed]:
        return False, f"Package '{package}' not in allowed packages list"
    
    # Map module to package name
    package_name = MODULE_TO_PACKAGE.get(package, package)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name, "-q", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode == 0:
            return True, f"✓ Installed {package_name}"
        else:
            return False, f"✗ Failed to install {package_name}: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, f"✗ Timeout installing {package_name}"
    except Exception as e:
        return False, f"✗ Error: {str(e)}"


def execute_in_docker(code: str, timeout: int = 60) -> dict:
    """Execute code in a Docker container (more secure)"""
    if not shutil.which("docker"):
        return {
            "success": False,
            "stdout": "",
            "stderr": "Docker not available. Set execution_mode to 'direct' or install Docker.",
            "return_code": -1,
            "execution_time": 0,
            "installed_packages": [],
        }
    
    image = CONFIG.get("docker_image", "python:3.12-slim")
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = f.name
    
    try:
        start = datetime.now()
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network=host",  # Allow network access
                "-v", f"{temp_path}:/code.py:ro",
                "--memory=512m",
                "--cpus=1",
                image,
                "python", "/code.py"
            ],
            capture_output=True, text=True, timeout=timeout + 10
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "execution_time": (datetime.now() - start).total_seconds(),
            "installed_packages": [],
            "mode": "docker",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "return_code": -1,
            "execution_time": timeout,
            "installed_packages": [],
            "mode": "docker",
        }
    finally:
        os.unlink(temp_path)


def execute_direct(
    code: str,
    timeout: int = 60,
    auto_install: bool = True,
    max_install_attempts: int = 3
) -> dict:
    """Execute code directly (faster, less isolated)"""
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "return_code": -1,
        "execution_time": 0.0,
        "installed_packages": [],
        "mode": "direct",
    }
    
    install_attempts = 0
    
    while install_attempts <= max_install_attempts:
        start = datetime.now()
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name
        
        try:
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(Path.home()),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["return_code"] = proc.returncode
            result["execution_time"] = (datetime.now() - start).total_seconds()
            
            if proc.returncode == 0:
                result["success"] = True
                break
            
            # Try auto-install
            if auto_install and install_attempts < max_install_attempts:
                missing = extract_missing_module(proc.stderr)
                if missing:
                    success, msg = install_package(missing)
                    if success:
                        result["installed_packages"].append(missing)
                        install_attempts += 1
                        continue
            
            break
            
        except subprocess.TimeoutExpired:
            result["stderr"] = f"Execution timed out after {timeout} seconds"
            break
        except Exception as e:
            result["stderr"] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            break
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    return result


def execute_code(
    code: str,
    timeout: int = None,
    auto_install: bool = None,
    mode: str = None
) -> dict:
    """Execute Python code using configured mode"""
    timeout = timeout or CONFIG.get("default_timeout", 60)
    auto_install = auto_install if auto_install is not None else CONFIG.get("auto_install", True)
    mode = mode or CONFIG.get("execution_mode", "direct")

    if mode == ExecutionMode.DOCKER.value:
        return execute_in_docker(code, timeout)
    else:
        return execute_direct(code, timeout, auto_install)


def execute_streaming(
    code: str,
    timeout: int = 60,
    auto_install: bool = True
) -> Generator[str, None, dict]:
    """
    Execute code with streaming output (yields lines as they're produced).

    Yields:
        Lines of output (stdout/stderr) as they're generated

    Returns:
        Final execution result dict
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "return_code": -1,
        "execution_time": 0.0,
        "installed_packages": [],
        "mode": "direct-stream",
    }

    max_install_attempts = 3
    install_attempts = 0
    stdout_lines = []
    stderr_lines = []

    while install_attempts <= max_install_attempts:
        start = datetime.now()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            # Start process with line-buffered output
            process = subprocess.Popen(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(Path.home()),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            # Platform-specific streaming
            if platform.system() == 'Windows':
                # Windows: Use threads for non-blocking I/O
                stdout_queue = queue.Queue()
                stderr_queue = queue.Queue()

                def enqueue_output(pipe, q):
                    for line in iter(pipe.readline, ''):
                        q.put(line)
                    pipe.close()

                stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
                stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
                stdout_thread.daemon = True
                stderr_thread.daemon = True
                stdout_thread.start()
                stderr_thread.start()

                # Read from queues
                while process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty():
                    try:
                        line = stdout_queue.get_nowait()
                        stdout_lines.append(line)
                        yield f"[OUT] {line.rstrip()}"
                    except queue.Empty:
                        pass

                    try:
                        line = stderr_queue.get_nowait()
                        stderr_lines.append(line)
                        yield f"[ERR] {line.rstrip()}"
                    except queue.Empty:
                        pass

                    # Check timeout
                    if (datetime.now() - start).total_seconds() > timeout:
                        process.kill()
                        yield f"[TIMEOUT] Execution exceeded {timeout}s"
                        break

            else:
                # Unix: Use select for non-blocking I/O
                import fcntl

                # Set non-blocking mode
                for pipe in [process.stdout, process.stderr]:
                    flags = fcntl.fcntl(pipe, fcntl.F_GETFL)
                    fcntl.fcntl(pipe, fcntl.F_SETFL, flags | os.O_NONBLOCK)

                # Stream output
                while True:
                    # Check if process is done
                    if process.poll() is not None:
                        # Drain remaining output
                        for line in process.stdout:
                            stdout_lines.append(line)
                            yield f"[OUT] {line.rstrip()}"
                        for line in process.stderr:
                            stderr_lines.append(line)
                            yield f"[ERR] {line.rstrip()}"
                        break

                    # Check timeout
                    if (datetime.now() - start).total_seconds() > timeout:
                        process.kill()
                        yield f"[TIMEOUT] Execution exceeded {timeout}s"
                        break

                    # Wait for data with timeout
                    readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                    for stream in readable:
                        line = stream.readline()
                        if line:
                            if stream == process.stdout:
                                stdout_lines.append(line)
                                yield f"[OUT] {line.rstrip()}"
                            else:
                                stderr_lines.append(line)
                                yield f"[ERR] {line.rstrip()}"

            # Get final status
            process.wait()
            result["return_code"] = process.returncode
            result["stdout"] = "".join(stdout_lines)
            result["stderr"] = "".join(stderr_lines)
            result["execution_time"] = (datetime.now() - start).total_seconds()

            if result["return_code"] == 0:
                result["success"] = True
                break

            # Try auto-install
            if auto_install and install_attempts < max_install_attempts:
                missing = extract_missing_module(result["stderr"])
                if missing:
                    yield f"[INSTALL] Detected missing module: {missing}"
                    success, msg = install_package(missing)
                    if success:
                        result["installed_packages"].append(missing)
                        yield f"[INSTALL] {msg}"
                        install_attempts += 1
                        stdout_lines = []
                        stderr_lines = []
                        continue

            break

        except Exception as e:
            result["stderr"] = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            yield f"[ERROR] {str(e)}"
            break
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    return result


def log_execution(code: str, description: str, result: dict):
    """Log execution for analytics"""
    try:
        logs = []
        if EXECUTION_LOG.exists():
            try:
                logs = json.loads(EXECUTION_LOG.read_text())
            except Exception:
                logs = []
        
        logs = logs[-999:]  # Keep last 1000
        
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            "code_lines": len(code.split("\n")),
            "success": result["success"],
            "execution_time": result["execution_time"],
            "mode": result.get("mode", "unknown"),
            "error_type": extract_error_type(result["stderr"]) if result["stderr"] else None,
        })
        
        EXECUTION_LOG.write_text(json.dumps(logs, indent=2))
    except Exception:
        pass


def extract_error_type(stderr: str) -> Optional[str]:
    """Extract error type from stderr"""
    match = re.search(r"(\w+Error|\w+Exception):", stderr)
    return match.group(1) if match else None

# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool()
def get_system_context() -> str:
    """
    Get comprehensive system context before writing code.
    
    CALL THIS FIRST to understand:
    - OS, Python version, available paths
    - Installed and available libraries
    - Execution mode (direct or Docker sandbox)
    - Past learnings from errors
    - Tips for writing effective code
    
    Returns detailed system information formatted for code generation.
    """
    return format_system_context()


# Internal execute function
def _execute_code_internal(
    code: str,
    timeout: int = 60,
    auto_install: bool = True,
    mode: str = None
) -> dict:
    """Internal execution dispatcher"""
    mode = mode or CONFIG.get("execution_mode", "direct")
    if mode == ExecutionMode.DOCKER.value:
        return execute_in_docker(code, timeout)
    return execute_direct(code, timeout, auto_install)


@mcp.tool()
def run_python(
    code: str,
    description: str = "",
    timeout: int = 60,
    auto_install: bool = True
) -> str:
    """
    Execute Python code to accomplish ANY task.
    
    This is a universal tool - write Python to do what you need:
    - HTTP requests (requests, httpx, aiohttp)
    - Parse HTML (beautifulsoup4, lxml)
    - Process data (pandas, json, csv)
    - File operations (pathlib, shutil)
    - System commands (subprocess)
    - Images (Pillow, opencv)
    - And anything else Python can do!
    
    Args:
        code: Python code to execute. Use print() for output.
        description: Brief task description (for logging)
        timeout: Max execution time in seconds
        auto_install: Auto-install missing packages
    
    Returns:
        Execution result with stdout, stderr, and status
    
    Example:
        code = '''
        import requests
        resp = requests.get("https://api.github.com/users/octocat")
        data = resp.json()
        print(f"User: {data['login']}")
        print(f"Repos: {data['public_repos']}")
        '''
    """
    result = _execute_code_internal(code, timeout, auto_install)
    
    # Format output
    parts = []
    
    if description:
        parts.append(f"📋 Task: {description}")
    
    status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
    parts.append(f"{status} | ⏱️ {result['execution_time']:.2f}s | Mode: {result.get('mode', 'direct')}")
    
    if result["installed_packages"]:
        parts.append(f"📦 Auto-installed: {', '.join(result['installed_packages'])}")
    
    parts.append("\n─── OUTPUT ───")
    parts.append(result["stdout"] if result["stdout"] else "(no output)")
    
    if result["stderr"]:
        parts.append("\n─── ERRORS ───")
        parts.append(result["stderr"])
    
    log_execution(code, description, result)
    
    return "\n".join(parts)


@mcp.tool()
def run_python_stream(
    code: str,
    description: str = "",
    timeout: int = 60,
    auto_install: bool = True
) -> str:
    """
    Execute Python code with REAL-TIME STREAMING OUTPUT.

    Perfect for long-running tasks where you want to see progress as it happens:
    - Web scraping multiple pages (see each page as it's scraped)
    - Data processing loops (see progress through large datasets)
    - API calls with retries (see each attempt)
    - File operations (see each file as it's processed)
    - Long computations (see intermediate results)

    Output streams in real-time as the code executes, so you see results
    immediately instead of waiting for the entire execution to complete.

    Args:
        code: Python code to execute. Use print() liberally for progress updates.
        description: Brief task description (for logging)
        timeout: Max execution time in seconds
        auto_install: Auto-install missing packages

    Returns:
        Streaming output followed by execution summary

    Example:
        code = '''
        import time
        for i in range(5):
            print(f"Processing item {i+1}/5...")
            time.sleep(1)
        print("✓ Done!")
        '''

    The output will appear line-by-line as the code runs, not all at once at the end.
    """
    # Collect all streamed output
    output_lines = []

    if description:
        output_lines.append(f"📋 Task: {description}")
        output_lines.append("🔄 Streaming output (real-time)...\n")

    # Stream execution
    start_time = datetime.now()
    result = None

    try:
        # Execute streaming - generator will yield lines and return result
        gen = execute_streaming(code, timeout, auto_install)

        try:
            while True:
                line = next(gen)
                output_lines.append(line)
        except StopIteration as e:
            # The return value is in e.value
            result = e.value

    except Exception as e:
        output_lines.append(f"\n❌ Streaming error: {str(e)}")
        result = {
            "success": False,
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "mode": "direct-stream",
            "installed_packages": [],
        }

    # Add summary
    output_lines.append("\n" + "─" * 60)

    if result and isinstance(result, dict):
        status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
        exec_time = result.get("execution_time", 0)
        mode = result.get("mode", "direct-stream")
        packages = result.get("installed_packages", [])

        output_lines.append(f"{status} | ⏱️ {exec_time:.2f}s | Mode: {mode}")

        if packages:
            output_lines.append(f"📦 Auto-installed: {', '.join(packages)}")

        # Log execution
        log_execution(code, description, result)
    else:
        output_lines.append("⏱️ Execution completed")

    return "\n".join(output_lines)


@mcp.tool()
def run_with_retry(
    code: str,
    description: str = "",
    max_retries: int = 3,
    timeout: int = 60
) -> str:
    """
    Execute Python code with intelligent retry and error analysis.
    
    On failure, this tool:
    1. Analyzes the error pattern
    2. Searches past learnings for solutions
    3. Provides diagnostic information
    4. Suggests fixes based on error type
    
    Use this for more robust execution when errors are expected.
    
    Args:
        code: Python code to execute
        description: Task description
        max_retries: Max retry attempts (same code)
        timeout: Execution timeout in seconds
    
    Returns:
        Detailed execution result with retry info and suggestions
    """
    attempts = []
    result = None
    
    for attempt in range(1, max_retries + 1):
        result = _execute_code_internal(code, timeout, auto_install=True)
        
        attempts.append({
            "attempt": attempt,
            "success": result["success"],
            "error": result["stderr"][:200] if result["stderr"] else None
        })
        
        if result["success"]:
            break
        
        # Don't retry identical errors
        if attempt > 1 and attempts[-1].get("error") == attempts[-2].get("error"):
            break
    
    # Format output
    parts = []
    
    if description:
        parts.append(f"📋 Task: {description}")
    
    status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
    parts.append(f"{status} | Attempts: {len(attempts)}/{max_retries}")
    parts.append(f"⏱️ {result['execution_time']:.2f}s | Mode: {result.get('mode', 'direct')}")
    
    if result["installed_packages"]:
        parts.append(f"📦 Auto-installed: {', '.join(result['installed_packages'])}")
    
    parts.append("\n─── OUTPUT ───")
    parts.append(result["stdout"] if result["stdout"] else "(no output)")
    
    if not result["success"] and result["stderr"]:
        parts.append("\n─── ERROR ANALYSIS ───")
        parts.append(result["stderr"])
        
        # Find relevant learnings
        relevant = learning_store.find_relevant(result["stderr"], code)
        if relevant:
            parts.append("\n💡 SUGGESTIONS FROM PAST LEARNINGS:")
            for i, learning in enumerate(relevant, 1):
                parts.append(f"   {i}. {learning.solution}")
                parts.append(f"      Context: {learning.context}")
        
        # Suggest based on error type
        error_type = extract_error_type(result["stderr"])
        if error_type:
            parts.append(f"\n🔍 Error type: {error_type}")
            if error_type == "ModuleNotFoundError":
                parts.append("   → Try: pip_install('package_name')")
            elif error_type == "FileNotFoundError":
                parts.append("   → Check: Does the file/path exist?")
            elif error_type == "PermissionError":
                parts.append("   → Try: Use /tmp for file operations")
    
    log_execution(code, description, result)
    
    return "\n".join(parts)


@mcp.tool()
def add_learning(
    error_pattern: str,
    solution: str,
    context: str,
    tags: str = ""
) -> str:
    """
    Record a learning from a code execution for future reference.
    
    When you figure out how to fix an error, record it here.
    Future executions will suggest this solution for similar errors.
    
    Args:
        error_pattern: Text/regex that matches the error message
        solution: What fixed the problem
        context: When this solution applies
        tags: Comma-separated tags (e.g., "network,ssl,https")
    
    Example:
        add_learning(
            error_pattern="SSL: CERTIFICATE_VERIFY_FAILED",
            solution="Add verify=False to requests.get() or install certifi",
            context="HTTPS requests on systems with certificate issues",
            tags="ssl,https,certificates"
        )
    
    Returns:
        Confirmation message
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    learning_store.add(error_pattern, solution, context, tag_list)
    
    return f"""✅ Learning recorded!

📝 Pattern: {error_pattern}
💡 Solution: {solution}
📌 Context: {context}
🏷️  Tags: {', '.join(tag_list) if tag_list else 'none'}

This will help with similar errors in future executions."""


@mcp.tool()
def get_learnings(search: str = "") -> str:
    """
    View recorded learnings from past executions.
    
    Args:
        search: Optional search term to filter learnings
    
    Returns:
        Summary of learnings, optionally filtered
    """
    if not search:
        return learning_store.get_summary()
    
    # Search learnings
    matches = []
    search_lower = search.lower()
    
    for l in learning_store.learnings:
        if (search_lower in l.error_pattern.lower() or 
            search_lower in l.solution.lower() or
            search_lower in l.context.lower() or
            any(search_lower in t.lower() for t in l.tags)):
            matches.append(l)
    
    if not matches:
        return f"No learnings found matching '{search}'"
    
    lines = [f"🔍 {len(matches)} learnings matching '{search}':\n"]
    for i, l in enumerate(matches, 1):
        lines.append(f"{i}. {l.error_pattern[:50]}...")
        lines.append(f"   → {l.solution}")
        lines.append(f"   Context: {l.context}")
        if l.tags:
            lines.append(f"   Tags: {', '.join(l.tags)}")
        lines.append("")
    
    return "\n".join(lines)


@mcp.tool()
def pip_install(package_name: str) -> str:
    """
    Pre-install a Python package.
    
    Use this to install packages before execution if you know
    you'll need them, or if auto-install missed something.
    
    Args:
        package_name: The pip package name to install
    
    Returns:
        Installation result
    """
    success, message = install_package(package_name)
    return message


@mcp.tool()
def configure(
    action: str = "view",
    key: str = "",
    value: str = ""
) -> str:
    """
    View or update Code Mode configuration.
    
    Args:
        action: "view" to see config, "set" to update a value
        key: Config key to update (for action="set")
        value: New value (for action="set")
    
    Available settings:
    - execution_mode: "direct" (fast) or "docker" (secure sandbox)
    - default_timeout: Default execution timeout in seconds
    - max_retries: Default max retry attempts
    - auto_install: Whether to auto-install packages (true/false)
    - docker_image: Docker image for sandbox mode
    
    Examples:
        configure()  # View current config
        configure(action="set", key="execution_mode", value="docker")
        configure(action="set", key="default_timeout", value="120")
    
    Returns:
        Current configuration or update confirmation
    """
    global CONFIG
    
    if action == "view":
        lines = ["⚙️  MCP Code Mode Configuration\n"]
        for k, v in CONFIG.items():
            lines.append(f"   {k}: {v}")
        lines.append(f"\n📁 Config file: {CONFIG_FILE}")
        lines.append(f"📚 Learnings: {len(learning_store.learnings)} recorded")
        return "\n".join(lines)
    
    elif action == "set":
        if not key:
            return "❌ Please specify a key to set"
        
        if key not in DEFAULT_CONFIG:
            return f"❌ Unknown config key: {key}\nValid keys: {', '.join(DEFAULT_CONFIG.keys())}"
        
        # Type conversion
        if key in ["default_timeout", "max_retries"]:
            try:
                value = int(value)
            except ValueError:
                return f"❌ {key} must be an integer"
        elif key == "auto_install":
            value = value.lower() in ["true", "1", "yes"]
        
        CONFIG[key] = value
        save_config(CONFIG)
        
        return f"✅ Updated {key} = {value}"
    
    return f"❌ Unknown action: {action}. Use 'view' or 'set'"


# ============================================================================
# Resources
# ============================================================================

@mcp.resource("codemode://context")
def context_resource() -> str:
    """System context as a resource"""
    return format_system_context()


@mcp.resource("codemode://learnings")
def learnings_resource() -> str:
    """All learnings as a resource"""
    return learning_store.get_summary()


@mcp.resource("codemode://config")
def config_resource() -> str:
    """Current configuration as a resource"""
    return json.dumps(CONFIG, indent=2)
