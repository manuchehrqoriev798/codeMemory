import numpy as np
import os
import faiss as faiss_lib
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaissIndex:
    """Vector similarity search index for code using CodeBERT embeddings."""
    
    def __init__(self, dimension=768):
        """Initialize FAISS index with CodeBERT for embeddings.
        
        Args:
            dimension (int): Embedding dimension for the index
        """
        self.index = faiss_lib.IndexFlatL2(dimension)  # L2 distance metric
        self.file_ids = []  # Store element IDs
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dimension = dimension
        self.element_info = {}  # Store metadata about indexed elements
        logger.info(f"Initialized FAISS index with dimension {dimension}")
        
    def get_embedding(self, content):
        """Generate an embedding for the given code content using CodeBERT.
        
        Args:
            content (str): Code content to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        inputs = self.tokenizer(content, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embedding
        
    def add_file(self, file_id, content):
        """Add a file's embedding to the FAISS index.
        
        Args:
            file_id: Unique identifier for the file
            content (str): File content to embed
        """
        embedding = self.get_embedding(content)
        self.index.add(embedding)
        self.file_ids.append(file_id)
        self.element_info[file_id] = {
            'type': 'file',
            'id': file_id,
            'content': content
        }
        logger.info(f"Added file {file_id} to FAISS index")
        
    def add_weighted_element(self, element_id, content, weights):
        """Add a code element with weighted embedding to the index.
        
        Args:
            element_id: Unique identifier for the element
            content (str): Element content to embed
            weights (dict): Weight values for different contexts
        """
        # Generate base embedding
        embedding = self.get_embedding(content)
        
        # Apply weights to the embedding
        weighted_embedding = embedding.copy()
        if weights:
            avg_weight = sum(weights.values()) / len(weights)
            weighted_embedding = weighted_embedding * avg_weight
        
        # Parse element ID to extract metadata
        element_id_str = str(element_id)
        element_info = {
            'id': element_id,
            'content': content,
            'weights': weights
        }
        
        # Extract element type and name from ID
        if "_function_" in element_id_str:
            element_info['type'] = "function"
            element_info['name'] = element_id_str.split("_function_")[1]
        elif "_method_" in element_id_str:
            element_info['type'] = "method"
            element_info['name'] = element_id_str.split("_method_")[1]
        elif "_class_" in element_id_str:
            element_info['type'] = "class"
            element_info['name'] = element_id_str.split("_class_")[1]
        else:
            element_info['type'] = "unknown"
            element_info['name'] = element_id_str.split("_")[-1] if "_" in element_id_str else element_id_str
        
        # Store metadata and add to index
        self.element_info[element_id] = element_info
        self.index.add(weighted_embedding)
        self.file_ids.append(element_id)
        logger.info(f"Added {element_info['type']} {element_info['name']} ({element_id}) to FAISS index")
        
    def search_similar(self, query_content, top_k=5, path_context=None):
        """Search for similar code elements using both embeddings and text matching.
        
        Args:
            query_content (str): The query text
            top_k (int): Number of results to return
            path_context (dict, optional): Path context weights
            
        Returns:
            list: List of element IDs of similar elements
        """
        # Generate query embedding
        query_embedding = self.get_embedding(query_content)
        
        # Apply path context weights if provided
        if path_context:
            avg_weight = sum(path_context.values()) / len(path_context)
            query_embedding = query_embedding * avg_weight
        
        # Reshape for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search for similar vectors
        D, I = self.index.search(query_embedding, min(top_k * 3, self.index.ntotal))
        
        # Create a list for all candidates with their scores
        results_with_scores = []
        
        # Add a boosted score for semantic similarity
        for i, idx in enumerate(I[0]):
            if idx < len(self.file_ids):
                element_id = self.file_ids[idx]
                
                # Get vector similarity score (invert distance to make it a similarity)
                vector_similarity = max(0, 100 - D[0][i])
                
                # Get element metadata
                element_info = self.element_info.get(element_id, {})
                element_type = element_info.get("type", "unknown")
                element_name = element_info.get("name", "")
                element_content = element_info.get("content", "")
                
                # Base score from vector similarity
                score = vector_similarity * 5  # Weight for semantic similarity
                
                # Boost score based on exact text match for element name
                # Give huge priority to exact matches
                if query_content.lower() == element_name.lower():
                    score += 10000  # Very high boost for exact matches
                elif query_content.lower() in element_name.lower():
                    # Partial match in name - boost based on how much of the name matches
                    match_ratio = len(query_content) / max(1, len(element_name))
                    score += 1000 * match_ratio
                
                # Boost score for text match in content
                if query_content.lower() in element_content.lower():
                    score += 200
                
                # Boost specific element types
                if element_type == "class":
                    score += 100
                elif element_type == "function":
                    score += 50
                elif element_type == "method":
                    score += 25
                
                results_with_scores.append((element_id, score, element_type, element_name))
        
        # Sort by score and return top results
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = [item[0] for item in results_with_scores[:top_k]]
        
        # Log top matches for debugging
        logger.debug(f"Top matches for query '{query_content}':")
        for i, (element_id, score, element_type, element_name) in enumerate(results_with_scores[:10]):
            logger.debug(f"{i+1}. {element_name} ({element_type}): {score} - {element_id}")
        
        return top_results
    
    def save(self, path="faiss_index.bin"):
        """Save the FAISS index to disk.
        
        Args:
            path (str): Path to save the index
        """
        faiss_lib.write_index(self.index, path)
        
        # Save file_ids and element_info
        with open(path + ".ids", "wb") as f:
            pickle.dump(self.file_ids, f)
            
        with open(path + ".metadata", "wb") as f:
            pickle.dump(self.element_info, f)
            
        logger.info(f"FAISS index saved to {path}")
    
    def load(self, path="faiss_index.bin"):
        """Load the FAISS index from disk.
        
        Args:
            path (str): Path to load the index from
        """
        if os.path.exists(path):
            self.index = faiss_lib.read_index(path)
            
            # Load file_ids
            if os.path.exists(path + ".ids"):
                with open(path + ".ids", "rb") as f:
                    self.file_ids = pickle.load(f)
            
            # Load element_info
            if os.path.exists(path + ".metadata"):
                with open(path + ".metadata", "rb") as f:
                    self.element_info = pickle.load(f)
                    
            logger.info(f"FAISS index loaded from {path} with {len(self.file_ids)} elements")
        else:
            logger.warning(f"Index file not found at {path}")
    
    def get_file_embedding(self, file_id):
        """Get the embedding for a specific element ID.
        
        Args:
            file_id: ID of the element to get the embedding for
            
        Returns:
            numpy.ndarray: Embedding vector or None if not found
        """
        if file_id in self.file_ids:
            idx = self.file_ids.index(file_id)
            return self.get_vector_by_index(idx)
        return None
    
    def get_vector_by_index(self, idx):
        """Get a vector from the index by its position.
        
        Args:
            idx (int): Index position
            
        Returns:
            numpy.ndarray: Vector at the specified position or None
        """
        if idx < self.index.ntotal:
            vector = np.zeros((1, self.dimension), dtype=np.float32)
            self.index.reconstruct(idx, vector[0])
            return vector[0]
        return None
    
    def get_element_info(self, element_id):
        """Get metadata about an indexed element.
        
        Args:
            element_id: ID of the element
            
        Returns:
            dict: Element metadata or None if not found
        """
        return self.element_info.get(element_id)
    
    def export_index_summary(self, output_path="faiss_index_summary.json"):
        """Export a summary of the index to a JSON file.
        
        Args:
            output_path (str): Path to save the summary
        """
        import json
        
        # Count by type
        type_counts = {}
        for element_id, info in self.element_info.items():
            element_type = info.get('type', 'unknown')
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        summary = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal,
            "total_elements": len(self.file_ids),
            "element_counts_by_type": type_counts,
            "file_ids": self.file_ids[:10] + ["..."] if len(self.file_ids) > 10 else self.file_ids
        }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Index summary exported to {output_path}")
        return summary