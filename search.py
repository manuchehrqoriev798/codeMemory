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
    
    def __init__(self, index_path="faiss_index.bin", 
                 neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", 
                 neo4j_password="password"):
        """Initialize the search engine.
        
        Args:
            index_path (str): Path to the FAISS index
            neo4j_uri (str): Neo4j connection URI
            neo4j_user (str): Neo4j username
            neo4j_password (str): Neo4j password
        """
        self.index_path = index_path
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
        """Get detailed information for elements from Neo4j.
        
        Args:
            element_ids (list): List of element IDs to retrieve
            
        Returns:
            list: Element details from Neo4j
        """
        self._connect_to_neo4j()
        
        element_details = []
        query = """
        MATCH (n)
        WHERE n.file_id IN $element_ids 
           OR n.function_id IN $element_ids
           OR n.class_id IN $element_ids
           OR n.method_id IN $element_ids
        RETURN COALESCE(n.file_id, n.function_id, n.class_id, n.method_id) as id,
               COALESCE(n.name, "Unknown") as name, 
               n.content as content,
               labels(n) as labels,
               n.file_path as file_path,
               n.line_number as line_number
        """
        
        # Collect records from Neo4j directly into a list of dictionaries
        records = []
        try:
            # Instead of using run_query, use session directly to ensure proper result handling
            with self.neo4j_conn.driver.session() as session:
                result = session.run(query, element_ids=element_ids)
                # Collect all records into a list of dictionaries
                for record in result:
                    record_dict = {
                        "id": record["id"],
                        "name": record["name"],
                        "content": record["content"],
                        "labels": record["labels"],
                        "file_path": record["file_path"],
                        "line_number": record["line_number"]
                    }
                    records.append(record_dict)
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            records = []
        
        # Now process these records
        for record in records:
            details = {
                "id": record["id"],
                "name": record["name"],
                "content": record["content"],
                "type": record["labels"][0] if record["labels"] else "Unknown",
                "file_path": record["file_path"],
                "line_number": record["line_number"]
            }
            element_details.append(details)
        
        # If some elements weren't found in Neo4j, use metadata from FAISS index
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
                        "line_number": None
                    }
                    element_details.append(details)
        
        return element_details
    
    def search(self, query, top_k=5, path_context=None):
        """Search for code elements matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            path_context (dict, optional): Path context weights
            
        Returns:
            dict: Search results with element details
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Connect to Neo4j if not already connected
        if self.neo4j_conn is None:
            self._connect_to_neo4j()
        
        # First, try to find exact matches in element names
        exact_matches = []
        
        # Look for exact matches in function, class, and method names
        # Note: We're collecting the results in a list to avoid ResultConsumedError
        cypher_query = """
        MATCH (e:Function)
        WHERE toLower(e.name) CONTAINS toLower($query_param)
        RETURN e.function_id AS id, e.name AS name, 'function' AS type, 
               e.file_path AS file_path, e.line_number AS line_number, 
               e.content AS content
        UNION
        MATCH (e:Class)
        WHERE toLower(e.name) CONTAINS toLower($query_param)
        RETURN e.class_id AS id, e.name AS name, 'class' AS type,
               e.file_path AS file_path, e.line_number AS line_number,
               e.content AS content
        UNION
        MATCH (e:Method)
        WHERE toLower(e.name) CONTAINS toLower($query_param)
        RETURN e.method_id AS id, e.name AS name, 'method' AS type,
               e.file_path AS file_path, e.line_number AS line_number,
               e.content AS content
        UNION
        MATCH (e:File)
        WHERE toLower(e.file_name) CONTAINS toLower($query_param)
        RETURN e.file_id AS id, e.file_name AS name, 'file' AS type,
               e.file_path AS file_path, null AS line_number,
               e.content AS content
        """
        
        # Collect the results directly into a list
        exact_records = []
        try:
            # Instead of using run_query which might auto-close the session
            with self.neo4j_conn.driver.session() as session:
                result = session.run(cypher_query, query_param=query)
                # Collect all records into a list
                for record in result:
                    exact_records.append(record)
        except Exception as e:
            logger.error(f"Error querying Neo4j for exact matches: {e}")
            exact_records = []
        
        # Process the exact matches
        for record in exact_records:
            element_id = record.get("id")
            if element_id:
                element_info = {
                    "id": element_id,
                    "name": record.get("name", "Unknown"),
                    "type": record.get("type", "Unknown"),
                    "file_path": record.get("file_path"),
                    "line_number": record.get("line_number"),
                    "content": record.get("content", "")
                }
                exact_matches.append(element_info)
        
        # Then get semantic matches using FAISS
        semantic_ids = self.faiss_index.search_similar(query, top_k + len(exact_matches), path_context)
        
        # Get detailed information for semantic matches
        semantic_matches = []
        exact_match_ids = [match["id"] for match in exact_matches]
        
        for element_id in semantic_ids:
            # Skip IDs that are already in exact matches
            if element_id in exact_match_ids:
                continue
            
            details = self._get_element_details([element_id])
            if details:
                semantic_matches.extend(details)
        
        # Combine results, prioritizing exact matches
        combined_results = exact_matches + semantic_matches
        
        # Limit to top_k results
        combined_results = combined_results[:top_k]
        
        # Format results
        results = []
        for i, details in enumerate(combined_results):
            element_id = details.get("id")
            embedding = self.faiss_index.get_file_embedding(element_id)
            embedding_preview = embedding[:10].tolist() if embedding is not None else []
            
            # Format content preview
            content = details.get("content", "")
            content_preview = content[:150] + "..." if len(content) > 150 else content
            
            result = {
                "rank": i + 1,
                "id": element_id,
                "name": details.get("name", "Unknown"),
                "type": details.get("type", "Unknown"),
                "file_path": details.get("file_path", "Unknown"),
                "line_number": details.get("line_number"),
                "content_preview": content_preview,
                "embedding_preview": embedding_preview,
                "match_type": "exact" if i < len(exact_matches) else "semantic"
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return {
            "query": query,
            "results": results,
            "total": len(results)
        }
    
    def save_results(self, results, output_path="search_results.json"):
        """Save search results to a JSON file.
        
        Args:
            results (dict): Search results
            output_path (str): Path to save results
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Search results saved to {output_path}")
    
    def get_index_summary(self):
        """Get summary information about the FAISS index.
        
        Returns:
            dict: Index summary
        """
        # Export summary
        summary = self.faiss_index.export_index_summary()
        return summary
    
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