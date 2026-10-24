import re
import subprocess
import sys
import tempfile
import pathlib
from typing import Tuple, Optional

def extract_code(answer: str) -> Optional[str]:
    """
    Extracts Python code from markdown code blocks in the LLM answer.
    Supports ```python ... ``` and generic ``` ... ``` blocks.
    """
    pattern = r"```(?:python)?\s*\n(.*?)\s*```"
    match = re.search(pattern, answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_code(code: str) -> Tuple[bool, str]:
    """
    Writes extracted code to a temporary file, executes it securely via subprocess 
    using the active Python interpreter, enforces a 5-second timeout, and returns (success, output).
    """
    if not code:
        return False, "No code block found to execute."

    # Write code to a secure temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # Run subprocess with sys.executable and a 5-second timeout
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout if result.stdout else "Executed successfully with no stdout."
            return True, output
        else:
            error_output = result.stderr if result.stderr else result.stdout
            return False, f"Execution failed with exit code {result.returncode}:\n{error_output}"
            
    except subprocess.TimeoutExpired:
        return False, "Execution timed out after 5 seconds."
    except Exception as e:
        return False, f"Execution error: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            pathlib.Path(temp_file_path).unlink(missing_ok=True)
        except Exception:
            pass