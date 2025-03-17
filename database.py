from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, password):
        """Initialize Neo4j connection with provided credentials."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def clear_database(self):
        """Remove all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_constraints(self):
        """Create unique constraints for all entity types."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.project_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:File) REQUIRE f.file_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Goal) REQUIRE g.goal_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.task_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Resource) REQUIRE r.resource_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Commit) REQUIRE c.commit_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (tr:Trace) REQUIRE tr.trace_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dependency) REQUIRE d.dependency_id IS UNIQUE")
            print("Constraints created.")

    def create_project(self, project_id, goal, language):
        """Create a new project node in Neo4j."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (p:Project {project_id: $project_id, goal: $goal, language: $language, created_at: datetime()})
                """,
                project_id=project_id, goal=goal, language=language
            )
            print(f"Project {project_id} created.")

    def create_file(self, file_id, file_name, file_path, content, project_id):
        """Create a file node and link it to a project."""
        with self.driver.session() as session:
            # Check if file already exists
            file_exists = session.run(
                "MATCH (f:File {file_id: $file_id}) RETURN f",
                file_id=file_id
            ).single()
            
            if file_exists:
                # File exists, just ensure it's linked to the project
                session.run(
                    """
                    MATCH (f:File {file_id: $file_id})
                    MATCH (p:Project {project_id: $project_id})
                    MERGE (f)-[:BELONGS_TO]->(p)
                    """,
                    file_id=file_id, project_id=project_id
                )
                print(f"File {file_id} already exists, ensured link to Project {project_id}.")
            else:
                # Create new file and link to project
                session.run(
                    """
                    CREATE (f:File {file_id: $file_id, file_name: $file_name, 
                                    file_path: $file_path, content: $content})
                    WITH f
                    MATCH (p:Project {project_id: $project_id})
                    CREATE (f)-[:BELONGS_TO]->(p)
                    """,
                    file_id=file_id, file_name=file_name, file_path=file_path, 
                    content=content, project_id=project_id
                )
                print(f"File {file_id} created and linked to Project {project_id}.")

    def create_goal(self, goal_id, description, status, priority, file_id=None):
        """Create a goal node, optionally linking it to a file."""
        with self.driver.session() as session:
            query = """
                CREATE (g:Goal {goal_id: $goal_id, description: $description, status: $status, priority: $priority})
                """
            if file_id:
                query += """
                    WITH g
                    MATCH (f:File {file_id: $file_id})
                    CREATE (g)-[:LINKS_TO]->(f)
                    """
            session.run(query, goal_id=goal_id, description=description, status=status, priority=priority, file_id=file_id)
            print(f"Goal {goal_id} created.")

    def get_file_by_id(self, file_id):
        """Retrieve a file by its ID."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (f:File {file_id: $file_id}) RETURN f",
                file_id=file_id
            )
            record = result.single()
            if record:
                return dict(record["f"])
            return None

    def run_query(self, query, **params):
        """Run an arbitrary query on Neo4j."""
        with self.driver.session() as session:
            return session.run(query, **params)

    def create_class_node(self, class_id, name, line_number, weights, file_id, file_path=None, content=None, description=None):
        """Create a class node and link it to a file."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (c:Class {
                    class_id: $class_id,
                    name: $name,
                    line_number: $line_number,
                    weights: $weights,
                    file_path: $file_path,
                    content: $content,
                    description: $description
                })
                WITH c
                MATCH (f:File {file_id: $file_id})
                CREATE (c)-[:BELONGS_TO]->(f)
                """,
                class_id=class_id,
                name=name,
                line_number=line_number,
                weights=str(weights),
                file_id=file_id,
                file_path=file_path,
                content=content,
                description=description
            )
            print(f"Class {class_id} created and linked to File {file_id}.")

    def create_function_node(self, function_id, name, line_number, weights, file_id, file_path=None, content=None, description=None, called_functions=None, called_by=None, input_params=None, return_type=None, calls_list=None):
        """Create a function node and link it to a file."""
        try:
            with self.driver.session() as session:
                # First ensure file exists and is connected to a project
                file_result = session.run(
                    """
                    MATCH (f:File {file_id: $file_id})
                    OPTIONAL MATCH (f)-[:BELONGS_TO]->(p:Project)
                    RETURN f, p
                    """,
                    file_id=file_id
                ).single()
                
                if not file_result:
                    print(f"Warning: File {file_id} not found when creating function {function_id}")
                
                # Create function node and link to file
                session.run(
                    """
                    CREATE (f:Function {
                        function_id: $function_id,
                        name: $name,
                        line_number: $line_number,
                        weights: $weights,
                        file_path: $file_path,
                        content: $content,
                        description: $description,
                        input_params: $input_params,
                        return_type: $return_type,
                        calls_list: $calls_list
                    })
                    WITH f
                    MATCH (file:File {file_id: $file_id})
                    CREATE (f)-[:DEFINED_IN]->(file)
                    """,
                    function_id=function_id,
                    name=name,
                    line_number=line_number,
                    weights=str(weights),
                    file_id=file_id,
                    file_path=file_path,
                    content=content,
                    description=description,
                    input_params=str(input_params) if input_params else None,
                    return_type=return_type,
                    calls_list=calls_list
                )
                print(f"Function {function_id} created and linked to File {file_id}.")

                # Process function calls with explicit CALL relationship creation
                if called_functions:
                    for called_function_name in called_functions:
                        # Create CALLS relationship with explicit query
                        session.run(
                            """
                            MATCH (source:Function {function_id: $function_id})
                            MATCH (target)
                            WHERE (target:Function AND target.name = $called_function_name) OR 
                                  (target:Method AND target.name = $called_function_name)
                            CREATE (source)-[:CALLS]->(target)
                            """,
                            function_id=function_id,
                            called_function_name=called_function_name
                        )
                        print(f"  Created CALLS relationship: {function_id} -> {called_function_name}")
        except Exception as e:
            print(f"Error creating function node: {e}")
            logger.error(f"Error creating function node: {e}")

    def create_method_node(self, method_id, name, line_number, weights, class_id, file_path=None, content=None, description=None, called_functions=None, called_by=None, input_params=None, return_type=None, calls_list=None):
        """Create a method node and link it to a class."""
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    CREATE (m:Method {
                        method_id: $method_id,
                        name: $name,
                        line_number: $line_number,
                        weights: $weights,
                        file_path: $file_path,
                        content: $content,
                        description: $description,
                        input_params: $input_params,
                        return_type: $return_type,
                        calls_list: $calls_list
                    })
                    WITH m
                    MATCH (c:Class {class_id: $class_id})
                    CREATE (m)-[:BELONGS_TO]->(c)
                    """,
                    method_id=method_id,
                    name=name,
                    line_number=line_number,
                    weights=str(weights),
                    class_id=class_id,
                    file_path=file_path,
                    content=content,
                    description=description,
                    input_params=str(input_params) if input_params else None,
                    return_type=return_type,
                    calls_list=calls_list
                )
                print(f"Method {method_id} created and linked to Class {class_id}.")

                # Process method calls with explicit CALL relationship creation
                if called_functions:
                    for called_function_name in called_functions:
                        # Create CALLS relationship with explicit query
                        session.run(
                            """
                            MATCH (source:Method {method_id: $method_id})
                            MATCH (target)
                            WHERE (target:Function AND target.name = $called_function_name) OR 
                                  (target:Method AND target.name = $called_function_name)
                            CREATE (source)-[:CALLS]->(target)
                            """,
                            method_id=method_id,
                            called_function_name=called_function_name
                        )
                        print(f"  Created CALLS relationship: {method_id} -> {called_function_name}")
        except Exception as e:
            print(f"Error creating method node: {e}")
            logger.error(f"Error creating method node: {e}")

    def get_code_element_by_id(self, element_id):
        """Retrieve a code element (function, class, method) by its ID."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e)
                WHERE e.function_id = $element_id OR e.class_id = $element_id OR e.method_id = $element_id
                RETURN e
                """,
                element_id=element_id
            )
            record = result.single()
            if record:
                return dict(record["e"])
            return None

    def rebuild_call_relationships(self):
        """Rebuild all CALLS relationships based on stored calls_list attributes."""
        with self.driver.session() as session:
            # First, remove all existing CALLS relationships
            session.run("MATCH ()-[r:CALLS]->() DELETE r")
            print("Removed all existing CALLS relationships")
            
            # Get all elements with calls_list
            result = session.run("""
                MATCH (n) 
                WHERE n.calls_list IS NOT NULL AND n.calls_list <> ''
                RETURN COALESCE(n.function_id, n.method_id) as id, n.calls_list as calls_list, labels(n)[0] as type
            """)
            
            count = 0
            for record in result:
                source_id = record["id"]
                source_type = record["type"]
                calls_list = record["calls_list"].split(',')
                
                for target_name in calls_list:
                    if target_name:
                        # Create CALLS relationship
                        session.run("""
                            MATCH (source) WHERE (source:Function AND source.function_id = $source_id) OR (source:Method AND source.method_id = $source_id)
                            MATCH (target) WHERE target.name = $target_name
                            MERGE (source)-[:CALLS]->(target)
                        """, source_id=source_id, target_name=target_name)
                        count += 1
            
            print(f"Rebuilt {count} CALLS relationships")
            return count

    def ensure_all_files_connected_to_project(self, project_id="project_code_memory"):
        """Ensure all files are connected to a project.
        
        This utility method will find any files not connected to any project
        and connect them to the specified project ID.
        """
        with self.driver.session() as session:
            # Find files not connected to any project
            result = session.run(
                """
                MATCH (f:File)
                WHERE NOT (f)-[:BELONGS_TO]->(:Project)
                RETURN f.file_id as file_id
                """,
            )
            
            orphaned_files = [record["file_id"] for record in result]
            
            if not orphaned_files:
                print("All files are already connected to projects.")
                return 0
            
            # Connect orphaned files to the specified project
            for file_id in orphaned_files:
                session.run(
                    """
                    MATCH (f:File {file_id: $file_id})
                    MATCH (p:Project {project_id: $project_id})
                    CREATE (f)-[:BELONGS_TO]->(p)
                    """,
                    file_id=file_id, project_id=project_id
                )
            
            print(f"Connected {len(orphaned_files)} orphaned files to Project {project_id}.")
            return len(orphaned_files)