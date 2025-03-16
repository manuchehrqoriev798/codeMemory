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
            session.run(
                """
                CREATE (f:File {file_id: $file_id, file_name: $file_name, file_path: $file_path, 
                                content: $content, last_updated: datetime()})
                WITH f
                MATCH (p:Project {project_id: $project_id})
                CREATE (f)-[:BELONGS_TO]->(p)
                """,
                file_id=file_id, file_name=file_name, file_path=file_path, content=content, project_id=project_id
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

    def create_class_node(self, class_id, name, line_number, weights, file_id):
        """Create a class node and link it to a file."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (c:Class {
                    class_id: $class_id,
                    name: $name,
                    line_number: $line_number,
                    weights: $weights
                })
                WITH c
                MATCH (f:File {file_id: $file_id})
                CREATE (c)-[:BELONGS_TO]->(f)
                """,
                class_id=class_id,
                name=name,
                line_number=line_number,
                weights=str(weights),
                file_id=file_id
            )
            print(f"Class {class_id} created and linked to File {file_id}.")

    def create_function_node(self, function_id, name, line_number, weights, file_id):
        """Create a function node and link it to a file."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (f:Function {
                    function_id: $function_id,
                    name: $name,
                    line_number: $line_number,
                    weights: $weights
                })
                WITH f
                MATCH (file:File {file_id: $file_id})
                CREATE (f)-[:BELONGS_TO]->(file)
                """,
                function_id=function_id,
                name=name,
                line_number=line_number,
                weights=str(weights),
                file_id=file_id
            )
            print(f"Function {function_id} created and linked to File {file_id}.")

    def create_method_node(self, method_id, name, line_number, weights, class_id):
        """Create a method node and link it to a class."""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (m:Method {
                    method_id: $method_id,
                    name: $name,
                    line_number: $line_number,
                    weights: $weights
                })
                WITH m
                MATCH (c:Class {class_id: $class_id})
                CREATE (m)-[:BELONGS_TO]->(c)
                """,
                method_id=method_id,
                name=name,
                line_number=line_number,
                weights=str(weights),
                class_id=class_id
            )
            print(f"Method {method_id} created and linked to Class {class_id}.")

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