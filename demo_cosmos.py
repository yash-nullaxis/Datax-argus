"""
Demo script for Argus SDK with Azure Cosmos DB (SQL API) using Agno.

This demonstrates how to use the Agno-based Argus implementation
to query an Azure Cosmos DB database using natural language.

Environment Variables Required:
    GOOGLE_API_KEY       - Your Google AI API key
    COSMOS_ENDPOINT      - Cosmos DB account endpoint (e.g. https://<account>.documents.azure.com:443/)
    COSMOS_KEY           - Cosmos DB account primary or secondary key
    COSMOS_DATABASE      - Cosmos DB database name

    Alternatively, you can provide:
    COSMOS_CONNECTION_STR - Full Cosmos DB connection string (overrides endpoint/key)
"""
import asyncio
import os
import argparse
from dotenv import load_dotenv

# Load .env to get API keys and Cosmos config
load_dotenv()

from argus import Orchestrator, AgentConfig, ModelConfig, CosmosDBConnector
from pydantic import SecretStr


async def main():
    parser = argparse.ArgumentParser(description="Argus SDK - Cosmos DB Demo (Agno)")
    parser.add_argument("--verbose", action="store_true", help="Enable aggressive logging")
    parser.add_argument("--query", type=str, default=None, help="Custom query to run")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # Cosmos DB connection details
    cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
    cosmos_key = os.getenv("COSMOS_KEY")
    cosmos_database = os.getenv("COSMOS_DATABASE")
    cosmos_connection_str = os.getenv("COSMOS_CONNECTION_STR")

    if not cosmos_database:
        print("Error: COSMOS_DATABASE not found in environment.")
        return

    if not cosmos_connection_str and (not cosmos_endpoint or not cosmos_key):
        print("Error: Provide either COSMOS_CONNECTION_STR or both COSMOS_ENDPOINT and COSMOS_KEY.")
        return

    # 1. Configuration for Gemini 2.0 Flash
    config = AgentConfig(
        model=ModelConfig(
            provider="google", 
            model_name="gemini-2.0-flash", 
            api_key=SecretStr(api_key)
        ),
        db_connection_str=cosmos_connection_str or f"{cosmos_endpoint}|{cosmos_database}",
        verbose=args.verbose,
        log_sql=True,
    )

    # 2. Setup Data Connector for Cosmos DB
    print(f"Connecting to Cosmos DB database: {cosmos_database}...")
    
    try:
        if cosmos_connection_str:
            connector = CosmosDBConnector(
                endpoint="",  # Not needed when using connection string
                key="",       # Not needed when using connection string
                database=cosmos_database,
                connection_str=cosmos_connection_str,
            )
        else:
            connector = CosmosDBConnector(
                endpoint=cosmos_endpoint,
                key=cosmos_key,
                database=cosmos_database,
            )
        
        # Detailed connectivity check
        print("Verifying connection and fetching schema...")
        info = connector.get_schema_info()
        
        # Attempt to get container count
        try:
            schema_dict = connector.get_schema_dict()
            container_count = len(schema_dict)
            print(f"Connection Successful! Found {container_count} containers.")
        except Exception:
            print("Connection Successful! (Could not enumerate container details)")

        print(f"Schema Info Preview:\n{info[:500]}...")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to Cosmos DB: {e}")
        print("Ensure you have 'azure-cosmos' installed: pip install azure-cosmos")
        print("Check your endpoint, key, and database name.")
        return

    # 3. Initialize Agent
    agent = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    query = args.query or """
    Show me a summary of all data in the database. 
    List all containers and the count of documents in each.
    """
    
    print(f"\n--- Running Cosmos DB Analysis with Gemini 2.0 Flash ---\nQuery: {query}")
    try:
        print("\n\n--- Final Memo (Streaming) ---")
        full_memo = ""
        # Use stream_run to get tokens as they are generated
        async for chunk in agent.stream_run(query):
            print(chunk, end="", flush=True)
            full_memo += chunk
            
        print("\n\n[Done]")
    except Exception as e:
        print(f"\n[Error] Agent execution failed during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
