from typing import List, Dict, Any, Optional
from .connector import DBConnector
import json


class NoSQLConnector(DBConnector):
    """
    Abstract base for NoSQL adapters.
    """
    def get_schema_info(self) -> str:
        # Default behavior: generic description
        return "NoSQL Database. Dynamic Schema."
    
    def get_schema_dict(self) -> Dict[str, List[str]]:
        return {}
        

class MongoDBConnector(NoSQLConnector):
    def __init__(self, connection_str: str, database: str):
        # Requires pymongo
        from pymongo import MongoClient
        self.client = MongoClient(connection_str)
        self.db = self.client[database]
        
    def get_schema_info(self) -> str:
        # List collections and get one sample from each to infer structure
        schema_parts = []
        for collection_name in self.db.list_collection_names():
            count = self.db[collection_name].count_documents({})
            # Get one document to sample structure
            sample = self.db[collection_name].find_one()
            if sample:
                # Convert ObjectId to str for display
                sample['_id'] = str(sample['_id'])
                sample_str = str(sample)
                fields = list(sample.keys())
            else:
                sample_str = "Empty Collection"
                fields = []
            
            schema_parts.append(f"Collection: {collection_name}\nDocs: {count}\nFields (Sampled): {', '.join(fields)}\nSample Document: {sample_str}")
            
        return "\n\n".join(schema_parts)

    def get_schema_dict(self) -> Dict[str, List[str]]:
        schema = {}
        for collection_name in self.db.list_collection_names():
            sample = self.db[collection_name].find_one()
            if sample:
                schema[collection_name] = [str(k) for k in sample.keys()]
            else:
                schema[collection_name] = []
        return schema

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        # Expects query to be a JSON string like: {"collection": "users", "filter": {"age": {"$gt": 25}}}
        # This requires the LLM to generate this specific JSON structure.
        try:
            q_dict = json.loads(query)
            collection = self.db[q_dict['collection']]
            filter_doc = q_dict.get('filter', {})
            limit = q_dict.get('limit', 10)
            
            cursor = collection.find(filter_doc).limit(limit)
            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id']) # Json serialize
                results.append(doc)
            return results
        except Exception as e:
            raise ValueError(f"Failed to execute MongoDB query. Ensure format is JSON {{'collection': '...', 'filter': {{...}}}}. Error: {e}")


class ChromaDBConnector(DBConnector):
    def __init__(self, client=None, collection_name: str = "documents"):
        # Abstracting ChromaDB. 
        # Typically used for retrieval, NOT "analytics" in the SQL sense.
        # But if we want to "query" it, we likely mean semantic search.
        import chromadb
        if client:
            self.client = client
        else:
            self.client = chromadb.Client() # Ephemeral by default
            
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Upsert documents into the collection.
        """
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def get_schema_info(self) -> str:
        count = self.collection.count()
        sample = self.collection.peek(limit=1)
        return f"Vector Collection: {self.collection_name}\nCount: {count}\nSample: {sample}"

    def get_schema_dict(self) -> Dict[str, List[str]]:
        return {self.collection_name: ["id", "content", "metadata"]}

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        # Query here implies semantic search string
        results = self.collection.query(query_texts=[query], n_results=5)
        # Parse results into list of dicts
        out = []
        ids = results['ids'][0]
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        for i in range(len(ids)):
            out.append({
                "id": ids[i],
                "content": docs[i],
                "metadata": metadatas[i]
            })
        return out
