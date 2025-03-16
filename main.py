import os
import ast
import re
from database import Neo4jConnection
from faiss_index import FaissIndex

class CodeAnalyzer:
    """Analyzes Python code to extract functions, classes and their relationships."""
    
    def __init__(self):
        self.functional_paths = {
            "data_processing": 0.8,
            "mathematical_operations": 0.9,
            "shape_calculations": 0.7,
            "validation": 0.6,
            "utility_functions": 0.5
        }
    
    def analyze_file(self, file_path, file_content):
        """Analyze a Python file and extract its structure with weights."""
        try:
            tree = ast.parse(file_content)
            
            # Store the tree for use in _save_code_structure_to_neo4j
            self.current_tree = tree
            
            # Extract all functions and classes
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'weights': self._calculate_weights(node, file_content),
                        'node': node  # Store the AST node
                    }
                    functions.append(function_info)
                
                elif isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'lineno': item.lineno,
                                'weights': self._calculate_weights(item, file_content),
                                'node': item  # Store the AST node
                            }
                            methods.append(method_info)
                    
                    class_info = {
                        'name': node.name,
                        'lineno': node.lineno,
                        'methods': methods,
                        'weights': self._calculate_weights(node, file_content),
                        'node': node  # Store the AST node
                    }
                    classes.append(class_info)
            
            return {
                'functions': functions,
                'classes': classes
            }
        
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return {'functions': [], 'classes': []}
    
    def _calculate_weights(self, node, content):
        """Calculate weights for a code element based on its content and purpose."""
        node_content = ast.get_source_segment(content, node)
        if not node_content:
            return {path: 0.1 for path in self.functional_paths}
        
        weights = {}
        
        # Assign weights based on content analysis
        for path, base_weight in self.functional_paths.items():
            weight = base_weight  # Start with base weight
            
            # Adjust weights based on keywords
            if path == "mathematical_operations":
                if re.search(r'math\.|np\.|calculate|compute|sum|average|mean|median', node_content):
                    weight *= 1.5
                
            elif path == "shape_calculations":
                if re.search(r'area|perimeter|volume|circle|rectangle|triangle|shape', node_content):
                    weight *= 1.5
                
            elif path == "validation":
                if re.search(r'valid|check|assert|raise|error|exception', node_content):
                    weight *= 1.5
                
            elif path == "data_processing":
                if re.search(r'data|process|transform|convert|parse', node_content):
                    weight *= 1.5
                
            weights[path] = min(weight, 1.0)  # Cap at 1.0
        
        return weights

class CodeProcessor:
    """Handles loading, analyzing, and storing code in Neo4j and FAISS."""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        """Initialize the processor with connections to Neo4j and FAISS."""
        self.neo4j = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        self.faiss_index = FaissIndex()
        self.analyzer = CodeAnalyzer()
        
        # Add necessary methods to Neo4j and FAISS classes
        self._extend_neo4j_and_faiss()
        
    def _extend_neo4j_and_faiss(self):
        """Add extended methods to Neo4j and FAISS classes."""
        # Add run_query method to Neo4jConnection if not present
        if not hasattr(Neo4jConnection, 'run_query'):
            def run_query(self, query, **params):
                """Run a query and properly handle session lifecycle"""
                with self.driver.session() as session:
                    return session.run(query, **params)
            Neo4jConnection.run_query = run_query
        
        # Add weighted_element method to FaissIndex if not present
        if not hasattr(FaissIndex, 'add_weighted_element'):
            def add_weighted_element(self, element_id, content, weights):
                """Add weighted element to FAISS index"""
                embedding = self.get_embedding(content)
                
                # Apply weights to the embedding
                weighted_embedding = embedding.copy()
                if weights:
                    avg_weight = sum(weights.values()) / len(weights)
                    weighted_embedding = weighted_embedding * avg_weight
                
                self.index.add(weighted_embedding)
                self.file_ids.append(element_id)
                print(f"Added element {element_id} to FAISS with weights.")
            FaissIndex.add_weighted_element = add_weighted_element

    def setup_database(self):
        """Set up Neo4j database with constraints and clear existing data."""
        self.neo4j.clear_database()
        self.neo4j.create_constraints()
        
        # Create additional constraints for code elements
        self.neo4j.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.class_id IS UNIQUE")
        self.neo4j.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.method_id IS UNIQUE")
        self.neo4j.run_query("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE f.function_id IS UNIQUE")
    
    def create_project(self, project_id, name, language):
        """Create a project in Neo4j."""
        self.neo4j.create_project(project_id, name, language)
        
    def process_file(self, file_path, file_id, project_id):
        """Process a single file and add it to Neo4j and FAISS."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"ERROR: File not found: {file_path}")
                return False
            
            # Read the file
            with open(file_path, "r") as f:
                file_content = f.read()
            
            # Store current file info for use in _save_code_structure_to_neo4j
            self.current_file_content = file_content
            self.current_file_path = file_path
            
            # Get file name from path
            file_name = os.path.basename(file_path)
            
            # Add file to Neo4j
            self.neo4j.create_file(file_id, file_name, file_path, file_content, project_id)
            
            # Analyze code structure
            structure = self.analyzer.analyze_file(file_path, file_content)
            
            # Save structure to Neo4j
            self._save_code_structure_to_neo4j(file_id, structure)
            
            # Add to FAISS
            self._add_weighted_embeddings_to_faiss(structure, file_content, file_id)
            
            print(f"Successfully processed {file_name}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False
    
    def _save_code_structure_to_neo4j(self, file_id, structure):
        """Save code structure with weights to Neo4j database."""
        # Add class nodes
        for cls in structure['classes']:
            class_id = f"{file_id}_class_{cls['name']}"
            # Get the full class content including body
            class_content = ast.get_source_segment(self.current_file_content, cls['node'])
            
            self.neo4j.run_query(
                """
                MERGE (c:Class {class_id: $class_id})
                ON CREATE SET c.name = $name, 
                             c.line_number = $lineno, 
                             c.weights = $weights,
                             c.content = $content,
                             c.file_path = $file_path
                WITH c
                MATCH (f:File {file_id: $file_id})
                MERGE (c)-[:BELONGS_TO]->(f)
                """,
                class_id=class_id,
                name=cls['name'],
                lineno=cls['lineno'],
                weights=str(cls['weights']),
                content=class_content,
                file_path=self.current_file_path,
                file_id=file_id
            )
            
            # Add methods
            for method in cls['methods']:
                method_id = f"{class_id}_method_{method['name']}"
                # Get the full method content including body
                method_content = ast.get_source_segment(self.current_file_content, method['node'])
                
                self.neo4j.run_query(
                    """
                    MERGE (m:Method {method_id: $method_id})
                    ON CREATE SET m.name = $name, 
                                 m.line_number = $lineno, 
                                 m.weights = $weights,
                                 m.content = $content,
                                 m.file_path = $file_path
                    WITH m
                    MATCH (c:Class {class_id: $class_id})
                    MERGE (m)-[:BELONGS_TO]->(c)
                    """,
                    method_id=method_id,
                    name=method['name'],
                    lineno=method['lineno'],
                    weights=str(method['weights']),
                    content=method_content,
                    file_path=self.current_file_path,
                    class_id=class_id
                )
        
        # Add function nodes
        for func in structure['functions']:
            func_id = f"{file_id}_function_{func['name']}"
            # Get the full function content including body
            func_content = ast.get_source_segment(self.current_file_content, func['node'])
            
            self.neo4j.run_query(
                """
                MERGE (f:Function {function_id: $func_id})
                ON CREATE SET f.name = $name, 
                             f.line_number = $lineno, 
                             f.weights = $weights,
                             f.content = $content,
                             f.file_path = $file_path
                WITH f
                MATCH (file:File {file_id: $file_id})
                MERGE (f)-[:BELONGS_TO]->(file)
                """,
                func_id=func_id,
                name=func['name'],
                lineno=func['lineno'],
                weights=str(func['weights']),
                content=func_content,
                file_path=self.current_file_path,
                file_id=file_id
            )
    
    def _add_weighted_embeddings_to_faiss(self, structure, file_content, file_id):
        """Add weighted embeddings for code elements to FAISS index."""
        # Add overall file embedding
        self.faiss_index.add_file(file_id, file_content)
        
        # Add embeddings for classes
        for cls in structure['classes']:
            class_id = f"{file_id}_class_{cls['name']}"
            class_content = f"class {cls['name']}:"
            self.faiss_index.add_weighted_element(class_id, class_content, cls['weights'])
            
            # Add embeddings for methods
            for method in cls['methods']:
                method_id = f"{class_id}_method_{method['name']}"
                method_content = f"def {method['name']}:"
                self.faiss_index.add_weighted_element(method_id, method_content, method['weights'])
        
        # Add embeddings for functions
        for func in structure['functions']:
            func_id = f"{file_id}_function_{func['name']}"
            func_content = f"def {func['name']}:"
            self.faiss_index.add_weighted_element(func_id, func_content, func['weights'])
    
    def search_similar(self, query, top_k=5, path_context=None):
        """Search for similar code elements."""
        return self.faiss_index.search_similar(query, top_k, path_context)
    
    def save_faiss_index(self, path="faiss_index.bin"):
        """Save the FAISS index to disk."""
        self.faiss_index.save(path)
    
    def close(self):
        """Close all connections."""
        self.neo4j.close()

def main():
    # Initialize the code processor
    processor = CodeProcessor()
    
    try:
        # Setup the database
        processor.setup_database()
        
        # Create a project
        processor.create_project(1, "Code Analysis Project", "Python")
        
        # Process files
        files_to_process = [
            {"path": os.path.join(os.getcwd(), "code1.py"), "id": 1, "project_id": 1},
            {"path": os.path.join(os.getcwd(), "code2.py"), "id": 2, "project_id": 1},
            {"path": os.path.join(os.getcwd(), "code3.py"), "id": 3, "project_id": 1},
            {"path": os.path.join(os.getcwd(), "code4.py"), "id": 4, "project_id": 1}
        ]
        
        # Process each file
        for file_info in files_to_process:
            processor.process_file(file_info["path"], file_info["id"], file_info["project_id"])
        
        # Save the FAISS index
        processor.save_faiss_index()
        
        # Perform sample searches
        print("\n=== Searching for mathematical operations ===")
        similar_ids = processor.search_similar("calculate factorial recursive function", top_k=5)
        print(f"Similar elements: {similar_ids}")
        
        print("\n=== Searching for shape calculations ===")
        similar_ids = processor.search_similar("calculate area of geometric shapes", top_k=5)
        print(f"Similar elements: {similar_ids}")
        
        print("\n=== Searching for data processing ===")
        similar_ids = processor.search_similar("process and clean data from CSV", top_k=5)
        print(f"Similar elements: {similar_ids}")
        
        print("\n=== Searching for machine learning ===")
        similar_ids = processor.search_similar("train machine learning model with cross validation", top_k=5)
        print(f"Similar elements: {similar_ids}")
        
        # Add a goal to track implementation
        processor.neo4j.create_goal(1, "Implement weighted code structure analysis", "completed", 1, 1)
    
    finally:
        # Close connections
        processor.close()

if __name__ == "__main__":
    main()