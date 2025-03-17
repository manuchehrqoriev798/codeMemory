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
    
    def __init__(self, neo4j_connection=None):
        """Initialize code analyzer with dynamic functional paths and keywords."""
        # Initialize connection to Neo4j for path storage
        self.neo4j = neo4j_connection
        
        # Initialize empty dictionaries for paths and keywords
        self.functional_paths = {}
        self.path_keywords = {}
        
        # Load paths from database if connection is provided
        if self.neo4j:
            self.load_paths_from_database()
        else:
            # Default fallback paths if no database connection
            self._initialize_default_paths()
        
        logger.info("CodeAnalyzer initialized with %d functional paths", len(self.functional_paths))
    
    def _initialize_default_paths(self):
        """Initialize with common default paths if database is not available."""
        self.functional_paths = {
            "data_processing": 0.5,
            "user_interface": 0.5,
            "networking": 0.5,
            "authentication": 0.5,
            "file_operations": 0.5,
            "error_handling": 0.5,
            "algorithms": 0.5
        }
        
        self.path_keywords = {
            "data_processing": ["data", "process", "transform", "parse", "filter", "map", "reduce"],
            "user_interface": ["ui", "display", "render", "view", "interface", "button", "input"],
            "networking": ["http", "request", "response", "api", "endpoint", "url", "fetch"],
            "authentication": ["auth", "login", "password", "token", "permission", "user"],
            "file_operations": ["file", "read", "write", "open", "close", "save", "load"],
            "error_handling": ["error", "exception", "try", "catch", "handle", "log", "debug"],
            "algorithms": ["sort", "search", "algorithm", "optimize", "compute", "calculate"]
        }
        
        logger.info("Initialized with default functional paths")
    
    def load_paths_from_database(self):
        """Load functional paths and keywords from Neo4j database."""
        try:
            with self.neo4j.driver.session() as session:
                # Query for paths
                result = session.run(
                    """
                    MATCH (p:FunctionalPath)
                    OPTIONAL MATCH (p)-[:BELONGS_TO]->(hub:FunctionalPathsHub)
                    RETURN p.name as name, p.weight as weight, p.keywords as keywords
                    """
                )
                
                # Reset paths
                self.functional_paths = {}
                self.path_keywords = {}
                
                # Process results
                for record in result:
                    path_name = record["name"]
                    weight = record["weight"]
                    keywords = record["keywords"].split(",") if record["keywords"] else []
                    
                    # Store in dictionaries
                    self.functional_paths[path_name] = weight
                    self.path_keywords[path_name] = keywords
                
                logger.info(f"Loaded {len(self.functional_paths)} functional paths from database")
                
                # If no paths were found, initialize defaults
                if not self.functional_paths:
                    logger.warning("No paths found in database, initializing defaults")
                    self._initialize_default_paths()
                    self._save_default_paths_to_database()
                else:
                    # Make sure there's a hub node and all paths are connected to it
                    session.run(
                        """
                        MERGE (hub:FunctionalPathsHub {name: 'Functional Paths'})
                        WITH hub
                        MATCH (p:FunctionalPath)
                        WHERE NOT (p)-[:BELONGS_TO]->(hub)
                        MERGE (p)-[:BELONGS_TO]->(hub)
                        """
                    )
                    logger.info("Ensured all functional paths are connected to hub")
        
        except Exception as e:
            logger.error(f"Error loading functional paths from database: {e}")
            self._initialize_default_paths()
    
    def _save_default_paths_to_database(self):
        """Save default paths to database."""
        try:
            if self.neo4j:
                # First ensure the hub node exists
                with self.neo4j.driver.session() as session:
                    session.run(
                        """
                        MERGE (hub:FunctionalPathsHub {name: 'Functional Paths'})
                        SET hub.updated_at = datetime()
                        """
                    )
                    logger.info("Created or updated FunctionalPathsHub node")
                
                # Add each path and connect to the hub
                for path_name, weight in self.functional_paths.items():
                    keywords = self.path_keywords.get(path_name, [])
                    self.add_functional_path(path_name, weight, keywords)
                
        except Exception as e:
            logger.error(f"Error saving default paths to database: {e}")
    
    def add_functional_path(self, path_name, weight, keywords=None):
        """Add or update a functional path.
        
        Args:
            path_name (str): Name of the functional path
            weight (float): Base weight for the path
            keywords (list): Keywords associated with this path
        
        Returns:
            bool: Success status
        """
        keywords = keywords or []
        # Update local dictionaries
        self.functional_paths[path_name] = weight
        self.path_keywords[path_name] = keywords
        
        # Update database if connected
        if self.neo4j:
            try:
                with self.neo4j.driver.session() as session:
                    # Create or update the path node and connect to hub
                    session.run(
                        """
                        MERGE (p:FunctionalPath {name: $name})
                        SET p.weight = $weight,
                            p.keywords = $keywords,
                            p.updated_at = datetime()
                        WITH p
                        MERGE (hub:FunctionalPathsHub {name: 'Functional Paths'})
                        MERGE (p)-[:BELONGS_TO]->(hub)
                        """,
                        name=path_name,
                        weight=weight,
                        keywords=",".join(keywords)
                    )
                    logger.info(f"Added/updated functional path: {path_name}")
                    return True
            except Exception as e:
                logger.error(f"Error saving functional path to database: {e}")
                return False
        return True
    
    def remove_functional_path(self, path_name):
        """Remove a functional path.
        
        Args:
            path_name (str): Name of the path to remove
            
        Returns:
            bool: Success status
        """
        # Remove from local dictionaries
        if path_name in self.functional_paths:
            del self.functional_paths[path_name]
        
        if path_name in self.path_keywords:
            del self.path_keywords[path_name]
        
        # Remove from database if connected
        if self.neo4j:
            try:
                with self.neo4j.driver.session() as session:
                    session.run(
                        """
                        MATCH (p:FunctionalPath {name: $name})
                        DETACH DELETE p
                        """,
                        name=path_name
                    )
                    logger.info(f"Removed functional path: {path_name}")
                    return True
            except Exception as e:
                logger.error(f"Error removing functional path from database: {e}")
                return False
        return True
    
    def get_functional_paths(self):
        """Get all functional paths.
        
        Returns:
            dict: Dictionary with path information
        """
        paths_info = {}
        for path_name in self.functional_paths:
            paths_info[path_name] = {
                "weight": self.functional_paths[path_name],
                "keywords": self.path_keywords.get(path_name, [])
            }
        return paths_info
    
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
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                analysis_result['content'] = file_content
                tree = ast.parse(file_content)

                # Extract all functions and classes
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_info = self._analyze_function(node, file_content)
                        if function_info:
                            analysis_result['functions'].append(function_info)
                    elif isinstance(node, ast.ClassDef):
                        class_info = self._analyze_class(node, file_content)
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
        
        # Extract input parameters
        params = self._extract_params(node)
        
        # Extract return type
        return_type = self._extract_return_type(node)
        
        # Extract function calls
        called_functions = self._extract_function_calls(node)

        return {
            'name': function_name,
            'code': code,
            'line_number': node.lineno,
            'params': params,
            'return_type': return_type,
            'weights': weights,
            'called_functions': called_functions
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
        """Extract function calls from a function definition."""
        called_functions = []
        
        # Walk through all nodes in the function body
        for child in ast.walk(node):
            # Look for function calls
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                # Add simple function name
                func_name = child.func.id
                if func_name not in called_functions and func_name != node.name:  # Avoid recursion
                    called_functions.append(func_name)
            # Handle method calls like obj.method()
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                # This is a method call like obj.method() - ignore for simple function relationships
                pass
        
        return called_functions

    def _extract_params(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
        """Extract parameters from a function definition with type annotations."""
        params = []
        
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': 'unknown'
            }
            
            # Extract type annotation if present
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_info['type'] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Subscript):
                    # Handle complex types like List[int]
                    try:
                        param_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                    except:
                        param_info['type'] = 'complex'
            
            params.append(param_info)
        
        return params

    def _extract_return_type(self, node: ast.FunctionDef) -> str:
        """Extract the return type annotation from a function definition."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Subscript):
                # Handle types like List[int]
                try:
                    return ast.unparse(node.returns) if hasattr(ast, 'unparse') else "complex"
                except:
                    return "complex"
            elif hasattr(ast, 'Constant') and isinstance(node.returns, ast.Constant) and node.returns.value is None:
                return "None"
        
        # Default return type if we can't determine it
        return "unknown"

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
