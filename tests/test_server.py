"""Tests for mcp-codemode server"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_codemode.server import (
    execute_direct,
    extract_missing_module,
    get_system_info,
    format_system_context,
    LearningStore,
    Learning,
    install_package,
    MODULE_TO_PACKAGE,
)


class TestCodeExecution:
    """Test code execution functionality"""
    
    def test_simple_print(self):
        """Test basic print statement"""
        result = execute_direct("print('Hello, World!')")
        assert result["success"] is True
        assert "Hello, World!" in result["stdout"]
        assert result["return_code"] == 0
    
    def test_math_calculation(self):
        """Test mathematical operations"""
        code = """
result = sum(range(1, 101))
print(f"Sum: {result}")
"""
        result = execute_direct(code)
        assert result["success"] is True
        assert "5050" in result["stdout"]
    
    def test_import_stdlib(self):
        """Test importing standard library"""
        code = """
import json
data = {"name": "test", "value": 42}
print(json.dumps(data))
"""
        result = execute_direct(code)
        assert result["success"] is True
        assert "test" in result["stdout"]
    
    def test_multiline_output(self):
        """Test multiple print statements"""
        code = """
for i in range(3):
    print(f"Line {i}")
"""
        result = execute_direct(code)
        assert result["success"] is True
        assert "Line 0" in result["stdout"]
        assert "Line 1" in result["stdout"]
        assert "Line 2" in result["stdout"]
    
    def test_syntax_error(self):
        """Test handling syntax errors"""
        result = execute_direct("print('unclosed")
        assert result["success"] is False
        assert "SyntaxError" in result["stderr"] or "EOL" in result["stderr"]
    
    def test_runtime_error(self):
        """Test handling runtime errors"""
        result = execute_direct("x = 1/0")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["stderr"]
    
    def test_name_error(self):
        """Test handling name errors"""
        result = execute_direct("print(undefined_variable)")
        assert result["success"] is False
        assert "NameError" in result["stderr"]
    
    def test_timeout(self):
        """Test execution timeout"""
        code = """
import time
time.sleep(10)
"""
        result = execute_direct(code, timeout=1)
        assert result["success"] is False
        assert "timed out" in result["stderr"].lower()
    
    def test_file_operations(self):
        """Test file read/write in temp directory"""
        code = """
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write("test content")
    path = f.name

with open(path, 'r') as f:
    content = f.read()

os.unlink(path)
print(f"Content: {content}")
"""
        result = execute_direct(code)
        assert result["success"] is True
        assert "test content" in result["stdout"]
    
    def test_pathlib_usage(self):
        """Test pathlib operations"""
        code = """
from pathlib import Path
import tempfile

tmp = Path(tempfile.gettempdir())
print(f"Temp exists: {tmp.exists()}")
print(f"Is dir: {tmp.is_dir()}")
"""
        result = execute_direct(code)
        assert result["success"] is True
        assert "True" in result["stdout"]


class TestModuleExtraction:
    """Test module name extraction from errors"""
    
    def test_no_module_named(self):
        """Test extracting from 'No module named' error"""
        error = "ModuleNotFoundError: No module named 'requests'"
        assert extract_missing_module(error) == "requests"
    
    def test_nested_module(self):
        """Test extracting base module from nested import"""
        error = "ModuleNotFoundError: No module named 'sklearn.ensemble'"
        assert extract_missing_module(error) == "sklearn"
    
    def test_double_quotes(self):
        """Test with double quotes"""
        error = 'ModuleNotFoundError: No module named "pandas"'
        assert extract_missing_module(error) == "pandas"
    
    def test_no_match(self):
        """Test when no module error is present"""
        error = "TypeError: 'int' object is not callable"
        assert extract_missing_module(error) is None
    
    def test_import_error(self):
        """Test ImportError format"""
        error = "ImportError: cannot import name 'foo' from 'bar'"
        assert extract_missing_module(error) == "bar"


class TestModuleMapping:
    """Test module to package name mapping"""
    
    def test_cv2_mapping(self):
        assert MODULE_TO_PACKAGE.get("cv2") == "opencv-python"
    
    def test_pil_mapping(self):
        assert MODULE_TO_PACKAGE.get("PIL") == "Pillow"
    
    def test_sklearn_mapping(self):
        assert MODULE_TO_PACKAGE.get("sklearn") == "scikit-learn"
    
    def test_yaml_mapping(self):
        assert MODULE_TO_PACKAGE.get("yaml") == "pyyaml"


class TestSystemInfo:
    """Test system information gathering"""
    
    def test_get_system_info(self):
        """Test system info retrieval"""
        info = get_system_info()
        assert "os" in info
        assert "python_version" in info
        assert "home" in info
        assert info["os"] in ["Linux", "Darwin", "Windows"]
    
    def test_has_python_executable(self):
        """Test Python executable is included"""
        info = get_system_info()
        assert "python_executable" in info
        assert "python" in info["python_executable"].lower()
    
    def test_format_context(self):
        """Test context formatting"""
        context = format_system_context()
        assert "SYSTEM CONTEXT" in context
        assert "AVAILABLE LIBRARIES" in context
        assert "Python" in context
        assert "EXECUTION NOTES" in context


class TestLearningStore:
    """Test the learning system"""
    
    def test_add_and_find_learning(self, tmp_path):
        """Test adding and finding learnings"""
        store = LearningStore(tmp_path / "learnings.json")
        
        store.add(
            error_pattern="SSL.*CERTIFICATE",
            solution="Use verify=False",
            context="HTTPS requests"
        )
        
        relevant = store.find_relevant("SSL: CERTIFICATE_VERIFY_FAILED")
        
        assert len(relevant) == 1
        assert "verify=False" in relevant[0].solution
    
    def test_persistence(self, tmp_path):
        """Test learnings persist across instances"""
        filepath = tmp_path / "learnings.json"
        
        store1 = LearningStore(filepath)
        store1.add("test_error", "test_solution", "test_context")
        
        store2 = LearningStore(filepath)
        assert len(store2.learnings) == 1
        assert store2.learnings[0].solution == "test_solution"
    
    def test_duplicate_updates_count(self, tmp_path):
        """Test duplicate patterns update success count"""
        store = LearningStore(tmp_path / "learnings.json")
        
        store.add("pattern", "solution1", "context")
        store.add("pattern", "solution2", "context")
        
        assert len(store.learnings) == 1
        assert store.learnings[0].success_count == 1
        assert store.learnings[0].solution == "solution2"
    
    def test_find_by_substring(self, tmp_path):
        """Test finding by substring match"""
        store = LearningStore(tmp_path / "learnings.json")
        
        store.add("connection refused", "check if server is running", "network errors")
        
        relevant = store.find_relevant("Error: connection refused to localhost:8080")
        assert len(relevant) == 1
    
    def test_tags(self, tmp_path):
        """Test learning tags"""
        store = LearningStore(tmp_path / "learnings.json")
        
        store.add(
            "timeout error",
            "increase timeout",
            "slow networks",
            tags=["network", "timeout"]
        )
        
        assert store.learnings[0].tags == ["network", "timeout"]
    
    def test_get_summary(self, tmp_path):
        """Test summary generation"""
        store = LearningStore(tmp_path / "learnings.json")
        store.add("error1", "fix1", "context1")
        store.add("error2", "fix2", "context2")
        
        summary = store.get_summary()
        assert "2 learnings" in summary
        assert "error1" in summary or "fix1" in summary


class TestAutoInstall:
    """Test auto-install functionality"""
    
    def test_auto_install_disabled(self):
        """Test that auto-install can be disabled"""
        code = "import some_nonexistent_package_xyz123"
        result = execute_direct(code, auto_install=False)
        assert result["success"] is False
        assert "ModuleNotFoundError" in result["stderr"]
        assert len(result["installed_packages"]) == 0


class TestExecutionModes:
    """Test different execution modes"""
    
    def test_direct_mode_default(self):
        """Test that direct mode is the default"""
        result = execute_direct("print('test')")
        assert result.get("mode") == "direct"
    
    def test_execution_time_tracked(self):
        """Test execution time is tracked"""
        result = execute_direct("print('fast')")
        assert "execution_time" in result
        assert result["execution_time"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
