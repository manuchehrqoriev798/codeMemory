import os
import json
import numpy as np
import torch
import logging
from faiss_index import FaissIndex
from database import Neo4jConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodeSearch:
    """Search engine for code elements using FAISS and Neo4j."""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", 
                 neo4j_password="password"):
        """Initialize the search engine."""
        self.index_path = "faiss_index.bin"
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Initialize search components
        self.faiss_index = FaissIndex()
        self.neo4j_conn = None
        
        # Load the FAISS index
        self._load_index()
        logger.info("Code search engine initialized")
        
    def _load_index(self):
        """Load the FAISS index."""
        if os.path.exists(self.index_path):
            self.faiss_index.load(self.index_path)
            logger.info(f"Loaded FAISS index with {self.faiss_index.index.ntotal} vectors")
        else:
            logger.warning(f"FAISS index not found at {self.index_path}")
    
    def _connect_to_neo4j(self):
        """Connect to Neo4j database."""
        if self.neo4j_conn is None:
            self.neo4j_conn = Neo4jConnection(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password
            )
            logger.info("Connected to Neo4j database")
    
    def _get_element_details(self, element_ids):
        """Get detailed information for elements from Neo4j."""
        self._connect_to_neo4j()
        
        element_details = []
        query = """
        MATCH (n)
        WHERE n.file_id IN $element_ids 
           OR n.function_id IN $element_ids
           OR n.class_id IN $element_ids
           OR n.method_id IN $element_ids
        OPTIONAL MATCH (n)-[:DEFINED_IN]->(f:File)
        RETURN COALESCE(n.function_id, n.class_id, n.method_id, n.file_id) as id,
               COALESCE(n.name, "Unknown") as name, 
               n.content as content,
               labels(n) as labels,
               COALESCE(n.file_path, f.file_path) as file_path,
               n.line_number as line_number,
               n.weights as weights,
               n.description as description
        """
        
        # Collect records from Neo4j
        records = []
        try:
            with self.neo4j_conn.driver.session() as session:
                result = session.run(query, element_ids=element_ids)
                for record in result:
                    record_dict = {
                        "id": record["id"],
                        "name": record["name"],
                        "content": record["content"],
                        "labels": record["labels"],
                        "file_path": record["file_path"],
                        "line_number": record["line_number"],
                        "weights": record["weights"],
                        "description": record["description"]
                    }
                    records.append(record_dict)
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            records = []
        
        # Process records
        for record in records:
            details = {
                "id": record["id"],
                "name": record["name"],
                "content": record["content"],
                "type": record["labels"][0] if record["labels"] else "Unknown",
                "file_path": record["file_path"],
                "line_number": record["line_number"],
                "weights": record["weights"],
                "description": record["description"]
            }
            element_details.append(details)
        
        # Add missing elements from FAISS metadata
        found_ids = {item["id"] for item in element_details}
        for element_id in element_ids:
            if element_id not in found_ids:
                element_info = self.faiss_index.get_element_info(element_id)
                if element_info:
                    details = {
                        "id": element_id,
                        "name": element_info.get("name", "Unknown"),
                        "content": element_info.get("content", ""),
                        "type": element_info.get("type", "Unknown"),
                        "file_path": None,
                        "line_number": None,
                        "weights": None,
                        "description": None
                    }
                    element_details.append(details)
        
        return element_details
    
    def search(self, query, top_k=5, focus_area=None, max_depth=3):
        """Search for code elements matching the query with context-aware traversal.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            focus_area (str): Functional path to focus on
            max_depth (int): Maximum traversal depth
            
        Returns:
            dict: Search results with adaptive detail
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k}, focus_area={focus_area})")
        
        # Connect to Neo4j if not already connected
        if self.neo4j_conn is None:
            self._connect_to_neo4j()
        
        # Get initial matches
        exact_matches = self._get_exact_matches(query)
        semantic_ids = self.faiss_index.search_similar(query, top_k + len(exact_matches), focus_area)
        
        # Get semantic matches excluding exact matches
        exact_match_ids = [match["id"] for match in exact_matches]
        semantic_matches = []
        for element_id in semantic_ids:
            if element_id not in exact_match_ids:
                details = self._get_element_details([element_id])
                if details:
                    semantic_matches.extend(details)
        
        # Combine results
        combined_results = exact_matches + semantic_matches
        combined_results = combined_results[:top_k]
        
        # Enhance with context-aware traversal if focus area specified
        if focus_area and combined_results:
            self._enhance_with_context_traversal(combined_results, focus_area, max_depth)
        
        # Format and return results
        results = self._format_results(combined_results, query, focus_area, len(exact_matches))
        return results
    
    def _enhance_with_context_traversal(self, results, focus_area, max_depth):
        """Enhance results with context-aware traversal based on focus area.
        
        Args:
            results (list): Initial search results
            focus_area (str): Functional path to focus on
            max_depth (int): Maximum traversal depth
        """
        for result in results:
            # Initialize energy and description coefficient
            initial_energy = 1.0
            initial_dc = 1.0
            
            # Get related functions with dynamic energy decay
            element_id = result.get("id")
            result["related_elements"] = []
            
            if element_id:
                self._traverse_related_elements(
                    element_id, 
                    result["related_elements"], 
                    focus_area, 
                    energy=initial_energy, 
                    dc=initial_dc, 
                    depth=0, 
                    max_depth=max_depth,
                    visited=set()
                )
    
    def _traverse_related_elements(self, element_id, related_elements, focus_area, energy, dc, depth, max_depth, visited):
        """Traverse related elements with dynamic energy decay.
        
        Args:
            element_id (str): Current element ID
            related_elements (list): List to store related elements
            focus_area (str): Functional path to focus on
            energy (float): Current energy level
            dc (float): Current description coefficient
            depth (int): Current traversal depth
            max_depth (int): Maximum traversal depth
            visited (set): Set of visited element IDs
        """
        # Stop if we've reached max depth, energy too low, or already visited
        if depth >= max_depth or energy < 0.2 or element_id in visited:
            return
        
        visited.add(element_id)
        
        # Get called functions
        called_elements = self._get_called_elements(element_id)
        
        for called in called_elements:
            called_id = called.get("id")
            if not called_id or called_id in visited:
                continue
            
            # Get functional path weights
            weights_str = called.get("weights", "{}")
            try:
                weights = eval(weights_str)  # Convert string representation to dict
            except:
                weights = {}
            
            # Enhanced Energy Decay Coefficient calculation
            # Calculate EDC based on path relevance to focus area
            if focus_area in weights:
                # Higher relevance = lower decay (more energy preserved)
                focus_weight = weights.get(focus_area, 0.1)
                
                # Enhanced formula: EDC is inversely proportional to the relevance
                # This means high relevance (high weight) = low decay
                # We use a logarithmic scale to better differentiate between weights
                # and add a small constant to avoid division by zero
                import math
                edc = 1.0 - (0.7 * math.log(1 + 9 * focus_weight))
                
                # Adjust based on depth - deeper traversal should decay faster
                depth_factor = 1.0 - (depth / (max_depth * 2))
                edc = max(0.1, edc * depth_factor)
            else:
                # If path has no relevance to focus area, decay quickly
                edc = 0.5 + (0.1 * depth)  # Higher depth = faster decay
            
            # Apply decay to energy and description coefficient
            new_energy = energy * edc
            new_dc = dc * edc
            
            # Calculate relevance score for sorting/filtering
            relevance = weights.get(focus_area, 0.1) if focus_area else sum(weights.values()) / max(1, len(weights))
            
            # Add to related elements with detail level based on DC
            detail_level = "high" if new_dc > 0.7 else "medium" if new_dc > 0.4 else "low"
            related_elements.append({
                "id": called_id,
                "name": called.get("name", "Unknown"),
                "type": called.get("type", "Unknown"),
                "relevance": relevance,
                "detail_level": detail_level,
                "energy_level": round(new_energy, 2),  # Add energy level for debugging/visualization
                "description": self._get_element_description(called, detail_level)
            })
            
            # Continue traversal
            self._traverse_related_elements(
                called_id,
                related_elements,
                focus_area,
                new_energy,
                new_dc,
                depth + 1,
                max_depth,
                visited.copy()
            )
    
    def _get_called_elements(self, element_id):
        """Get elements called by the given element.
        
        Args:
            element_id (str): Element ID
            
        Returns:
            list: Called elements
        """
        self._connect_to_neo4j()
        called_elements = []
        
        query = """
        MATCH (source)-[:CALLS]->(target)
        WHERE source.function_id = $element_id OR source.method_id = $element_id
        RETURN target
        """
        
        try:
            with self.neo4j_conn.driver.session() as session:
                result = session.run(query, element_id=element_id)
                for record in result:
                    target = record["target"]
                    element_info = dict(target)
                    
                    # Determine element type
                    element_type = "Unknown"
                    if "function_id" in element_info:
                        element_type = "Function"
                        element_info["id"] = element_info["function_id"]
                    elif "method_id" in element_info:
                        element_type = "Method"
                        element_info["id"] = element_info["method_id"]
                    
                    element_info["type"] = element_type
                    called_elements.append(element_info)
        except Exception as e:
            logger.error(f"Error getting called elements: {e}")
        
        return called_elements
    
    def _get_element_description(self, element, detail_level):
        """Get element description with adaptive detail level.
        
        Args:
            element (dict): Element information
            detail_level (str): Detail level (high, medium, low)
            
        Returns:
            str: Element description
        """
        description = element.get("description")
        content = element.get("content", "")
        
        if not description:
            if detail_level == "high":
                # Full description
                return f"Full content: {content}"
            elif detail_level == "medium":
                # Abbreviated description
                return f"Summary: {content[:100]}..." if len(content) > 100 else content
            else:
                # Minimal info
                return f"Function {element.get('name', 'Unknown')}"
        
        return description
    
    def _format_results(self, results, query, focus_area=None, exact_match_count=0):
        """Format search results with adaptive detail."""
        formatted_results = []
        
        for i, details in enumerate(results):
            element_id = details.get("id")
            
            # Handle None content safely
            content = details.get("content") or ""
            # Ensure we're showing at least a reasonable amount of content
            content_preview = content[:500] + "..." if len(content) > 500 else content
            
            result = {
                "rank": i + 1,
                "id": element_id,
                "name": details.get("name", "Unknown"),
                "type": details.get("type", "Unknown"),
                "file_path": details.get("file_path", "Unknown"),
                "line_number": details.get("line_number"),
                "content_preview": content_preview,
                "match_type": "exact" if i < exact_match_count else "semantic"
            }
            
            # Add related elements if present
            if "related_elements" in details:
                result["related_elements"] = details["related_elements"]
            
            formatted_results.append(result)
        
        return {
            "query": query,
            "focus_area": focus_area,
            "results": formatted_results,
            "total": len(formatted_results)
        }
    
    def _get_exact_matches(self, query):
        """Get exact matches for a query.
        
        Args:
            query (str): Search query
            
        Returns:
            list: Exact matches
        """
        exact_matches = []
        
        cypher_query = """
        MATCH (e)
        WHERE (e:Function OR e:Class OR e:Method)
        AND toLower(e.name) CONTAINS toLower($query_param)
        RETURN DISTINCT
            COALESCE(e.function_id, e.class_id, e.method_id) AS id,
            e.name AS name,
            labels(e)[0] as type,
            e.file_path AS file_path,
            e.line_number AS line_number,
            e.content AS content,
            e.weights AS weights
        """
        
        try:
            with self.neo4j_conn.driver.session() as session:
                exact_records = list(session.run(cypher_query, query_param=query))
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            exact_records = []
        
        for record in exact_records:
            element_id = record.get("id")
            if element_id:
                element_info = {
                    "id": element_id,
                    "name": record.get("name", "Unknown"),
                    "type": record.get("type", "Unknown"),
                    "file_path": record.get("file_path"),
                    "line_number": record.get("line_number"),
                    "content": record.get("content", ""),
                    "weights": record.get("weights", "{}")
                }
                exact_matches.append(element_info)
        
        return exact_matches
    
    def save_results(self, results, output_path="search_results.json"):
        """Save search results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Search results saved to {output_path}")
    
    def get_index_summary(self):
        """Get summary information about the FAISS index."""
        return self.faiss_index.export_index_summary()
    
    def close(self):
        """Close all connections."""
        if self.neo4j_conn:
            self.neo4j_conn.close()
            logger.info("Closed Neo4j connection")

def interactive_search():
    """Run an interactive search session."""
    print("\n===== Code Search Engine =====\n")
    
    search_engine = CodeSearch()
    
    try:
        # Show index summary
        summary = search_engine.get_index_summary()
        print("\nIndex Summary:")
        print(f"Dimension: {summary['dimension']}")
        print(f"Total vectors: {summary['total_vectors']}")
        print(f"Elements by type: {summary['element_counts_by_type']}")
        
        while True:
            print("\n" + "="*50)
            query = input("Enter code to search for (or 'exit' to quit): ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting search. Goodbye!")
                break
            
            if not query.strip():
                print("Please enter a valid search query.")
                continue
            
            top_k = 5
            try:
                k_input = input(f"Number of results to show (default: {top_k}): ")
                if k_input.strip():
                    top_k = int(k_input)
            except ValueError:
                print(f"Invalid number, using default ({top_k})")
            
            print(f"\nSearching for: {query}")
            results = search_engine.search(query, top_k)
            
            # Save results to file
            search_engine.save_results(results)
            
            # Print search results
            print("\nSearch Results:")
            if not results["results"]:
                print("No matches found.")
            else:
                for result in results["results"]:
                    print(f"\nRank {result['rank']}: {result['name']} ({result['type']})")
                    print(f"File: {result['file_path']}")
                    if result['line_number']:
                        print(f"Line: {result['line_number']}")
                    print(f"Preview: {result['content_preview']}")
                    print("-" * 40)
            
            print("\nResults saved to search_results.json")
    
    finally:
        search_engine.close()

if __name__ == "__main__":
    interactive_search() 