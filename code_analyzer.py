import os
import ast
import re
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """Analyzes Python code to extract functions, classes and their relationships."""
    
    def __init__(self):
        """Initialize code analyzer with functional paths and keywords."""
        # Define functional paths for classification with base weights
        self.functional_paths = {
            "data_processing": 0.8,
            "mathematical_operations": 0.9,
            "validation": 0.6,
            "utility_functions": 0.5
        }
        
        # Keywords associated with each path for automatic classification
        self.path_keywords = {
            "data_processing": ["data", "process", "clean", "transform", "filter", "map", "reduce"],
            "mathematical_operations": ["calc", "compute", "math", "average", "sum", "multiply", "divide"],
            "validation": ["valid", "check", "verify", "assert", "ensure", "sanitize"],
            "utility_functions": ["util", "helper", "format", "convert", "parse"]
        }
        
        logger.info("CodeAnalyzer initialized with %d functional paths", len(self.functional_paths))
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a Python file and extract its structure with path weights.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Dictionary containing extracted functions and classes with metadata
        """
        analysis_result = {
            'file_id': f"file_{os.path.basename(file_path).replace('.', '_')}",
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'classes': [],
            'functions': [],
            'content': '',
            'error': None
        }

        try:
            with open(file_path, 'r') as file:
                content = file.read()
                analysis_result['content'] = content
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_info = self._analyze_function(node, content)
                        if function_info:
                            analysis_result['functions'].append(function_info)
                    elif isinstance(node, ast.ClassDef):
                        class_info = self._analyze_class(node, content)
                        if class_info:
                            analysis_result['classes'].append(class_info)

            logger.info(f"Successfully analyzed file: {file_path}")
            logger.info(f"Successfully analyzed file with {len(analysis_result['functions'])} functions and {len(analysis_result['classes'])} classes")
            return analysis_result

        except FileNotFoundError:
            analysis_result['error'] = f"File not found: {file_path}"
            logger.error(analysis_result['error'])
        except SyntaxError as e:
            analysis_result['error'] = f"Syntax error in {file_path}: {e}"
            logger.error(analysis_result['error'])
        except Exception as e:
            analysis_result['error'] = f"Error analyzing {file_path}: {e}"
            logger.error(analysis_result['error'])

        return analysis_result

    def _analyze_function(self, node: ast.FunctionDef, file_content: str) -> Optional[Dict[str, Any]]:
        """Analyze a function node."""
        function_name = node.name
        code_lines = ast.get_source_segment(file_content, node).splitlines()
        code = '\n'.join(code_lines)
        weights = self._calculate_path_weights(code)
        params = self._extract_params(node)
        called_functions = self._extract_function_calls(node) # Extract called functions

        return {
            'name': function_name,
            'code': code,
            'line_number': node.lineno,
            'params': params,
            'weights': weights,
            'called_functions': called_functions # Add called functions to analysis
        }

    def _analyze_class(self, node: ast.ClassDef, file_content: str) -> Optional[Dict[str, Any]]:
        """Analyze a class node."""
        class_name = node.name
        code_lines = ast.get_source_segment(file_content, node).splitlines()
        code = '\n'.join(code_lines)
        weights = self._calculate_path_weights(code)
        methods = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item, file_content)
                if method_info:
                    methods.append(method_info)

        return {
            'name': class_name,
            'code': code,
            'line_number': node.lineno,
            'methods': methods,
            'weights': weights
        }

    def _calculate_path_weights(self, code_snippet: str) -> Dict[str, float]:
        """Calculate weights for functional paths based on code content."""
        weights = {}
        code_words = re.findall(r'\b\w+\b', code_snippet) # Extract words
        if not code_words:
            return weights # Return empty weights if no words found

        for path, keywords in self.path_keywords.items():
            weight = self.functional_paths.get(path, 0.1) # Base weight
            for keyword in keywords:
                # Check if keyword is in code words (faster check)
                if keyword.lower() in [word.lower() for word in code_words]:
                    weight += 0.2
                # Slower, more thorough check in full code snippet
                if re.search(r'\b' + re.escape(keyword) + r'\b', code_snippet, re.IGNORECASE):
                    weight += 0.1

            if weight > self.functional_paths.get(path, 0.1): # Only add if weight increased
                 weights[path] = min(weight, 1.0) # Cap at 1.0

        return weights

    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within a function node."""
        called_functions = []
        for child_node in ast.walk(node):
            if isinstance(child_node, ast.Call) and isinstance(child_node.func, ast.Name):
                called_functions.append(child_node.func.id)
        return called_functions

    def _extract_params(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Extract function parameters with types if available."""
        params = []

        for arg in node.args.args:
            param = {'name': arg.arg, 'type': None}

            # Check for type annotation
            if hasattr(arg, 'annotation') and arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param['type'] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    # For complex types like List[int]
                    param['type'] = "complex"  # Simplifying complex types

            params.append(param)

        return params

    def _get_node_source(self, node: ast.AST, source: str) -> str:
        """Extract source code for a node."""
        try:
            # Get line numbers
            start_line = node.lineno
            
            # For Python 3.8+, we can use end_lineno
            if hasattr(node, 'end_lineno'):
                end_line = node.end_lineno
            else:
                # For older Python versions, estimate end line
                end_line = start_line
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        end_line = max(end_line, child.lineno)
            
            # Extract source lines
            lines = source.splitlines()
            
            # Ensure we don't go beyond the end of the file
            end_line = min(end_line, len(lines))
            
            # Extract the code
            return '\n'.join(lines[start_line-1:end_line])
            
        except Exception as e:
            logger.error("Error extracting node source: %s", str(e))
            return "# Error extracting source"
    
    def _calculate_weights(self, node: ast.AST, node_text: str) -> Dict[str, float]:
        """Calculate functional path weights for a node based on its code content."""
        weights = {}
        
        # Extract words from code and lowercase them
        code_words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', node_text.lower())
        
        # Calculate weights based on keyword presence
        for path, keywords in self.path_keywords.items():
            weight = 0
            for keyword in keywords:
                # Check if any code word contains the keyword
                for word in code_words:
                    if keyword.lower() in word:
                        weight += 0.2
                        break
            
            # Also check for full keyword phrases
            for keyword in keywords:
                if keyword.lower() in node_text.lower():
                    weight += 0.3
            
            # Store weight if significant
            if weight > 0:
                weights[path] = min(weight, 1.0)  # Cap at 1.0
        
        # Add default minimal weights for all paths
        for path in self.functional_paths:
            if path not in weights:
                weights[path] = 0.1
        
        return weights 