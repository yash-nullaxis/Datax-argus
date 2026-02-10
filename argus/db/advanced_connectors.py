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


class CosmosDBConnector(NoSQLConnector):
    """
    Connector for Azure Cosmos DB using the native SQL (Core) API.
    
    Requires the `azure-cosmos` package.
    
    Usage:
        connector = CosmosDBConnector(
            endpoint="https://<account>.documents.azure.com:443/",
            key="<primary-or-secondary-key>",
            database="mydb"
        )
    
    The LLM is expected to generate queries as JSON strings:
        {
            "container": "users",
            "query": "SELECT * FROM c WHERE c.age > @age",
            "parameters": [{"name": "@age", "value": 25}],
            "limit": 10
        }
    """

    def __init__(
        self,
        endpoint: str,
        key: str,
        database: str,
        connection_str: Optional[str] = None,
    ):
        """
        Initialize CosmosDBConnector.

        Args:
            endpoint: The Cosmos DB account endpoint URL 
                      (e.g. https://<account>.documents.azure.com:443/).
            key: The Cosmos DB account primary or secondary key.
            database: The name of the database to connect to.
            connection_str: Optional. If provided, endpoint and key are extracted
                            from the connection string automatically.
        """
        from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions
        self._cosmos_exceptions = cosmos_exceptions

        if connection_str:
            self.client = CosmosClient.from_connection_string(connection_str)
        else:
            self.client = CosmosClient(endpoint, credential=key)

        self.database_name = database
        self.db = self.client.get_database_client(database)

    def _list_containers(self) -> List[str]:
        """List all container names in the database."""
        return [c["id"] for c in self.db.list_containers()]

    def _sample_items(self, container_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Fetch a small sample of items from a container."""
        container = self.db.get_container_client(container_name)
        query = f"SELECT TOP {limit} * FROM c"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        # Remove Cosmos system properties for cleaner display
        cleaned = []
        for item in items:
            cleaned_item = {
                k: v for k, v in item.items()
                if not k.startswith("_")
            }
            cleaned.append(cleaned_item)
        return cleaned

    def get_schema_info(self) -> str:
        """
        Introspect all containers and return schema information inferred from
        sample documents, including document count, field names, and sample data.
        """
        schema_parts = []
        for container_name in self._list_containers():
            container = self.db.get_container_client(container_name)

            # Get approximate document count
            try:
                count_result = list(container.query_items(
                    query="SELECT VALUE COUNT(1) FROM c",
                    enable_cross_partition_query=True,
                ))
                doc_count = count_result[0] if count_result else 0
            except Exception:
                doc_count = "Unknown"

            # Sample documents to infer fields
            samples = self._sample_items(container_name, limit=3)
            if samples:
                # Union of all keys across samples for a richer schema view
                all_fields = set()
                for s in samples:
                    all_fields.update(s.keys())
                fields_str = ", ".join(sorted(all_fields))
                sample_str = str(samples[0])
            else:
                fields_str = "Empty Container"
                sample_str = "No documents"

            schema_parts.append(
                f"Container: {container_name}\n"
                f"Documents: {doc_count}\n"
                f"Fields (Sampled): {fields_str}\n"
                f"Sample Document: {sample_str}"
            )

        return "\n\n".join(schema_parts) if schema_parts else "No containers found."

    def get_schema_dict(self) -> Dict[str, List[str]]:
        """
        Return a mapping of container names to field lists inferred from sampling.
        """
        schema = {}
        for container_name in self._list_containers():
            samples = self._sample_items(container_name, limit=1)
            if samples:
                schema[container_name] = sorted(samples[0].keys())
            else:
                schema[container_name] = []
        return schema

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cosmos DB SQL API query.

        Expected query format (JSON string):
            {
                "container": "myContainer",
                "query": "SELECT * FROM c WHERE c.status = @status",
                "parameters": [{"name": "@status", "value": "active"}],
                "limit": 100
            }

        If the query string is a plain SQL string (not JSON), it will be
        executed directly against the first container found, or raise an error.
        """
        try:
            q_dict = json.loads(query)
        except json.JSONDecodeError:
            # If not valid JSON, treat the entire string as a raw SQL query.
            # Requires at least one container to exist.
            containers = self._list_containers()
            if not containers:
                raise ValueError("No containers found in database and query is not JSON-formatted.")
            q_dict = {"container": containers[0], "query": query}

        container_name = q_dict.get("container")
        sql_query = q_dict.get("query", "SELECT * FROM c")
        parameters = q_dict.get("parameters")
        limit = q_dict.get("limit")

        if not container_name:
            raise ValueError(
                "Query must specify a 'container'. "
                "Format: {\"container\": \"name\", \"query\": \"SELECT ...\"}"
            )

        container = self.db.get_container_client(container_name)

        # Apply limit via TOP if not already present in the query
        if limit and "TOP" not in sql_query.upper():
            sql_query = sql_query.replace("SELECT", f"SELECT TOP {limit}", 1)

        query_kwargs = {
            "query": sql_query,
            "enable_cross_partition_query": True,
        }
        if parameters:
            query_kwargs["parameters"] = parameters

        items = list(container.query_items(**query_kwargs))

        # Clean out Cosmos system fields (_rid, _self, _etag, _attachments, _ts)
        results = []
        for item in items:
            cleaned = {k: v for k, v in item.items() if not k.startswith("_")}
            results.append(cleaned)

        return results


class CosmosDBMongoConnector(NoSQLConnector):
    """
    Connector for Azure Cosmos DB using the MongoDB-compatible API.

    This is essentially a MongoDB connector pointed at a Cosmos DB endpoint
    with the MongoDB wire protocol. Use this when your Cosmos DB account is
    configured with the MongoDB API.

    Usage:
        connector = CosmosDBMongoConnector(
            connection_str="mongodb://<account>:<key>@<account>.mongo.cosmos.azure.com:10255/?ssl=true&...",
            database="mydb"
        )

    The LLM is expected to generate queries in the same JSON format as the
    MongoDBConnector:
        {
            "collection": "users",
            "filter": {"age": {"$gt": 25}},
            "projection": {"name": 1, "age": 1},
            "sort": {"age": -1},
            "limit": 10
        }
    """

    def __init__(self, connection_str: str, database: str):
        """
        Initialize CosmosDBMongoConnector.

        Args:
            connection_str: The MongoDB-compatible connection string for Cosmos DB.
                            Typically looks like:
                            mongodb://<account>:<key>@<account>.mongo.cosmos.azure.com:10255/
                            ?ssl=true&retrywrites=false&replicaSet=globaldb&maxIdleTimeMS=120000
                            &appName=@<account>@
            database: The name of the database to connect to.
        """
        from pymongo import MongoClient
        self.client = MongoClient(connection_str)
        self.db = self.client[database]
        self.database_name = database

    def get_schema_info(self) -> str:
        """
        Introspect all collections and return field information inferred
        from sample documents.
        """
        schema_parts = []
        for collection_name in self.db.list_collection_names():
            count = self.db[collection_name].count_documents({})
            sample = self.db[collection_name].find_one()
            if sample:
                sample["_id"] = str(sample["_id"])
                sample_str = str(sample)
                fields = list(sample.keys())
            else:
                sample_str = "Empty Collection"
                fields = []

            schema_parts.append(
                f"Collection: {collection_name}\n"
                f"Docs: {count}\n"
                f"Fields (Sampled): {', '.join(fields)}\n"
                f"Sample Document: {sample_str}"
            )

        return "\n\n".join(schema_parts) if schema_parts else "No collections found."

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
        """
        Execute a MongoDB-compatible query against Cosmos DB.

        Expected format (JSON string):
            {
                "collection": "users",
                "filter": {"status": "active"},
                "projection": {"name": 1, "email": 1},
                "sort": {"created_at": -1},
                "limit": 10
            }
        """
        try:
            q_dict = json.loads(query)
            collection_name = q_dict.get("collection")
            if not collection_name:
                raise ValueError("Query must include a 'collection' field.")

            collection = self.db[collection_name]
            filter_doc = q_dict.get("filter", {})
            projection = q_dict.get("projection")
            sort_spec = q_dict.get("sort")
            limit = q_dict.get("limit", 10)

            cursor = collection.find(filter_doc, projection)
            if sort_spec:
                cursor = cursor.sort(list(sort_spec.items()))
            cursor = cursor.limit(limit)

            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except json.JSONDecodeError:
            raise ValueError(
                "Failed to parse query. Expected JSON format: "
                "{\"collection\": \"...\", \"filter\": {...}}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to execute Cosmos DB Mongo query. "
                f"Ensure format is JSON {{\"collection\": \"...\", \"filter\": {{...}}}}. "
                f"Error: {e}"
            )


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
