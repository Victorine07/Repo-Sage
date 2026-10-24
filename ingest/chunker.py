import ast
import pathlib
from typing import List, Dict, Union

def parse_python_file(file_path: Union[str, pathlib.Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Parses a single Python file and returns a list of dictionaries containing 
    metadata and source code for all public functions.
    """
    path = pathlib.Path(file_path)
    if not path.exists():
        return []
    
    try:
        source_code = path.read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(source_code, filename=str(path))
    except SyntaxError:
        # Skip files with syntax errors
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private helper functions starting with '_'
            if node.name.startswith('_'):
                continue
            
            docstring = ast.get_docstring(node) or ""
            
            # Extract source code segment using ast module or fallback line slicing
            source = ast.get_source_segment(source_code, node)
            if source is None:
                lines = source_code.splitlines()
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                source = "\n".join(lines[start:end])

            chunks.append({
                "name": node.name,
                "file": str(path),
                "line": node.lineno,
                "docstring": docstring,
                "source": source
            })
            
    return chunks


def parse_directory(dir_path: Union[str, pathlib.Path]) -> List[Dict[str, Union[str, int]]]:
    """
    Recursively walks a directory using pathlib, parses all .py files,
    skips any 'tests' subdirectories, and returns combined function chunks.
    """
    root_path = pathlib.Path(dir_path)
    all_chunks = []
    
    if not root_path.exists():
        return all_chunks

    for file_path in root_path.rglob("*.py"):
        # Skip files located within any 'tests' directory
        if "tests" in file_path.parts:
            continue
            
        file_chunks = parse_python_file(file_path)
        all_chunks.extend(file_chunks)
        
    return all_chunks