import os
import logging
import asyncio
from typing import Dict, Any, Optional
from command_parser import CommandParser, confirm_action
from code_analyzer import CodeAnalyzer
from database import Neo4jConnection
from faiss_index import FaissIndex
from code_search import CodeSearch
from ai_descriptor import AIDescriptor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeMemoryAI:
    """Main class for the CodeMemory AI system."""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", neo4j_password="password"):
        """Initialize CodeMemory AI with all its components."""
        # Create a default project if it doesn't exist
        self.default_project_id = "project_code_memory"
        
        self.command_parser = CommandParser()
        self.code_analyzer = CodeAnalyzer()
        self.neo4j = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
        
        # Ensure default project exists
        self._create_default_project()
        
        self.faiss_index = FaissIndex()
        self.code_search = CodeSearch(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        self.ai_descriptor = AIDescriptor()
        
        # Initialize database
        self._setup_database()
        logger.info("CodeMemory AI initialized successfully")
    
    def _create_default_project(self):
        """Create a default project if it doesn't exist."""
        try:
            with self.neo4j.driver.session() as session:
                # Check if project exists
                result = session.run(
                    "MATCH (p:Project {project_id: $project_id}) RETURN p",
                    project_id=self.default_project_id
                )
                
                # If project doesn't exist, create it
                if not result.single():
                    session.run(
                        """
                        CREATE (p:Project {
                            project_id: $project_id, 
                            name: 'CodeMemory Default Project', 
                            created_at: datetime(),
                            description: 'Default project for CodeMemory AI'
                        })
                        """,
                        project_id=self.default_project_id
                    )
                    logger.info(f"Created default project {self.default_project_id}")
        except Exception as e:
            logger.error(f"Error creating default project: {e}")
            raise
    
    def _setup_database(self):
        """Set up the Neo4j database with necessary constraints."""
        try:
            self.neo4j.create_constraints()
            logger.info("Database initialized with constraints")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    async def process_add_file(self, file_path: str) -> bool:
        """Process a file addition command."""
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return False
        
        # Ask for analysis preference
        print(f"\nI'm about to ingest {file_path}.")
        print("\nPlease choose how to handle descriptions:")
        print("1. [Enter] - Automatically generate all descriptions")
        print("2. [a] - Ask for confirmation for each description")
        print("3. [m] - Manually write all descriptions")
        print("4. [n] - No descriptions, process faster")
        print("5. [c] - Cancel")
        
        choice = input("\nYour choice [Enter/a/m/n/c]: ").lower().strip()
        
        if choice == 'c':
            print("File addition cancelled.")
            return False
        elif choice == 'n':
            auto_analyze = False
            confirm_each = False
            manual_mode = False
            no_description_mode = True
        elif choice == 'a':
            auto_analyze = False
            confirm_each = True
            manual_mode = False
            no_description_mode = False
        elif choice == 'm':
            auto_analyze = False
            confirm_each = False
            manual_mode = True
            no_description_mode = False
        else:
            auto_analyze = True
            confirm_each = False
            manual_mode = False
            no_description_mode = False
        
        try:
            # Analyze the file
            analysis = self.code_analyzer.analyze_file(file_path)
            
            if 'error' in analysis and analysis['error'] is not None:
                print(f"Error analyzing file: {analysis['error']}")
                return False
            
            # Generate file ID and store in Neo4j
            file_id = analysis['file_id']
            self.neo4j.create_file(
                file_id=file_id,
                file_name=os.path.basename(file_path),
                file_path=os.path.abspath(file_path),
                content=analysis['content'],
                project_id=self.default_project_id  # Use the default project ID
            )
            
            # Process each code element
            for cls in analysis['classes']:
                class_id = f"{file_id}_class_{cls['name']}"
                description = None
                
                if not no_description_mode:
                    if auto_analyze:
                        description = await self.ai_descriptor.describe_code_element(cls['code'], 'class')
                    elif confirm_each:
                        print(f"\nGenerating description for class '{cls['name']}'...")
                        description = await self.ai_descriptor.describe_code_element(cls['code'], 'class')
                        print(f"Generated: '{description}'")
                        if not confirm_action("Accept this description? [Enter=yes/n=no]: "):
                            description = input("Enter new description: ")
                    elif manual_mode:
                        print(f"\nClass '{cls['name']}':")
                        print(cls['code'])
                        description = input("Enter description: ")
                
                # Store class in Neo4j with all attributes
                self.neo4j.create_class_node(
                    class_id=class_id,
                    name=cls['name'],
                    content=cls['code'],
                    line_number=cls.get('lineno', 0),
                    weights=cls.get('weights', {}),
                    file_id=file_id,
                    file_path=os.path.abspath(file_path),
                    description=description
                )
                
                # Process methods
                for method in cls['methods']:
                    method_id = f"{class_id}_method_{method['name']}"
                    description = None
                    
                    if not no_description_mode:
                        if auto_analyze:
                            description = await self.ai_descriptor.describe_code_element(method['code'], 'method')
                        elif confirm_each:
                            print(f"\nGenerating description for method '{method['name']}' in class '{cls['name']}'...")
                            description = await self.ai_descriptor.describe_code_element(method['code'], 'method')
                            print(f"Generated: '{description}'")
                            if not confirm_action("Accept this description? [Enter=yes/n=no]: "):
                                description = input("Enter new description: ")
                        elif manual_mode:
                            print(f"\nMethod '{method['name']}' in Class '{cls['name']}':")
                            print(method['code'])
                            description = input("Enter description: ")
                    
                    # Include relationship information in the database
                    input_params = method.get('params', [])
                    return_type = method.get('return_type', 'unknown')
                    called_functions = method.get('called_functions', [])
                    
                    # Store method with additional info
                    self.neo4j.create_method_node(
                        method_id=method_id,
                        name=method['name'],
                        content=method['code'],
                        line_number=method.get('lineno', 0),
                        weights=method.get('weights', {}),
                        class_id=class_id,
                        file_path=os.path.abspath(file_path),
                        description=description,
                        input_params=input_params,
                        return_type=return_type,
                        called_functions=called_functions
                    )
            
            # Process standalone functions
            for func in analysis['functions']:
                func_id = f"{file_id}_function_{func['name']}"
                description = None
                
                if not no_description_mode:
                    if auto_analyze:
                        description = await self.ai_descriptor.describe_code_element(func['code'], 'function')
                    elif confirm_each:
                        print(f"\nGenerating description for function '{func['name']}'...")
                        description = await self.ai_descriptor.describe_code_element(func['code'], 'function')
                        print(f"Generated: '{description}'")
                        if not confirm_action("Accept this description? [Enter=yes/n=no]: "):
                            description = input("Enter new description: ")
                    elif manual_mode:
                        print(f"\nFunction '{func['name']}':")
                        print(func['code'])
                        description = input("Enter description: ")
                
                # Include relationship information in the database
                input_params = func.get('params', [])
                return_type = func.get('return_type', 'unknown')
                called_functions = func.get('called_functions', [])
                
                # Store function with additional info
                self.neo4j.create_function_node(
                    function_id=func_id,
                    name=func['name'],
                    content=func['code'],
                    line_number=func.get('lineno', 0),
                    weights=func.get('weights', {}),
                    file_id=file_id,
                    file_path=os.path.abspath(file_path),
                    description=description,
                    input_params=input_params,
                    return_type=return_type,
                    called_functions=called_functions
                )
            
            print(f"\nFile {file_path} processed successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            print(f"Error processing file: {str(e)}")
            return False
    
    def process_describe_command(self, element_id: str) -> None:
        """Process a describe command."""
        element_data = self.code_search.get_element_details([element_id])
        if element_data and element_data['results']:
            element = element_data['results'][0]
            description = self.ai_descriptor.get_confirmed_description(
                element['content_preview'], element['type'], element['name']
            )
            if description:
                self.neo4j.update_element_description(element_id, description, element['type'])
                print(f"\nDescription updated for {element['type']} '{element['name']}'.")
            else:
                print("\nDescription generation failed.")
        else:
            print(f"Element with ID '{element_id}' not found.")
    
    def process_search_command(self, query: str) -> None:
        """Process a search command."""
        results = self.code_search.search(query)
        self._display_search_results(results)
    
    def process_help_command(self) -> None:
        """Process a help command by displaying help text."""
        help_text = self.command_parser.get_help_text()
        print(f"\n{help_text}")
    
    def process_unknown_command(self, text: str) -> None:
        """Process an unknown command."""
        print(f"Unknown command: {text}")
        print("Type 'help' to see available commands.")
    
    def process_command(self, command: str) -> bool:
        """Parse and process a command."""
        parsed_command = self.command_parser.parse(command)
        
        command_type = parsed_command['command']
        parameters = parsed_command
        
        if command_type == 'add_file':
            file_path = parameters.get('file_path')
            if file_path:
                asyncio.run(self.process_add_file(file_path))
            else:
                print("Error: File path missing for 'add' command.")
        elif command_type == 'describe':
            element_id = parameters.get('element_id')
            if element_id:
                self.process_describe_command(element_id)
            else:
                print("Error: Element ID missing for 'describe' command.")
        elif command_type == 'search':
            query = parameters.get('query')
            if query:
                self.process_search_command(query)
            else:
                print("Error: Search query missing for 'search' command.")
        elif command_type == 'help':
            self.process_help_command()
        elif command_type == 'quit':
            return False
        elif command_type == 'unknown':
            self.process_unknown_command(parameters.get('text', ''))
        
        return True
    
    def _display_search_results(self, results: Dict[str, Any]):
        """Display search results in a formatted way."""
        if not results.get('results'):
            print("No matches found.")
            return
        
        print(f"\nFound {results['total']} results for '{results['query']}':")
        for result in results['results']:
            print(f"\n{result['rank']}. {result['name']} ({result['type']})")
            print(f"File: {result['file_path']}")
            if result.get('line_number'):
                print(f"Line: {result['line_number']}")
            print(f"Preview: {result['content_preview']}")
            print("-" * 40)
    
    def run(self):
        """Run the interactive command loop."""
        print("\nWelcome to CodeMemory AI")
        print("Type 'help' for available commands")
        
        try:
            async def main_loop():
                while True:
                    command = input("\nEnter command: ").strip()
                    if not command:
                        continue
                    
                    if not await asyncio.to_thread(self.process_command, command):
                        print("Goodbye!")
                        break
            asyncio.run(main_loop())
            
        finally:
            self.neo4j.close()

def main():
    """Main entry point for the application."""
    try:
        app = CodeMemoryAI()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())