"""
Demo script for Argus SDK with PostgreSQL using Agno.

This demonstrates how to use the Agno-based Argus implementation
to query a PostgreSQL database using natural language.
"""
import asyncio
import os
import argparse
from dotenv import load_dotenv

# Load .env to get API key and other config
load_dotenv()

from argus import Orchestrator, AgentConfig, ModelConfig, SqlAlchemyConnector
from pydantic import SecretStr


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable aggressive logging")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # 1. Configuration for Gemini 2.0 Flash
    config = AgentConfig(
        model=ModelConfig(
            provider="google", 
            model_name="gemini-2.0-flash", 
            api_key=SecretStr(api_key)
        ),
        # This DSN is passed here but also used explicitly in connector below
        db_connection_str="postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres",
        verbose=args.verbose,
        log_sql=True,
    )

    # 2. Setup Data Connector for Postgres
    postgres_dsn = "postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres"
    print(f"Connecting to Postgres: {postgres_dsn}...")
    
    try:
        connector = SqlAlchemyConnector(connection_str=postgres_dsn)
        
        # Detailed connectivity check
        print("Verifying connection and fetching schema...")
        info = connector.get_schema_info()
        
        # Attempt to get precise table count
        try:
            schema_dict = connector.get_schema_dict()
            table_count = len(schema_dict)
            print(f"Connection Successful! Found {table_count} tables.")
        except Exception:
            print("Connection Successful! (Could not enumerate tables details)")

        print(f"Schema Info Preview:\n{info[:500]}...")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to connect to Postgres: {e}")
        print("Ensure you have 'psycopg2-binary' installed: pip install psycopg2-binary")
        print("Check your database URL and ensure the server is running.")
        return

    # 3. Initialize Agent
    agent = Orchestrator(config, connector=connector)

    # 4. Run Analysis
    # A generic query that works on any schema
    query = """
For following key routes, can you comapre frequency of Indigo, Air India and AIX for red eye flights. Red eye flights are departing between 10pm and 4am. Routes to be considered Del-BOM BOM-DEL BLR-DEL DEL-BLR BLR-BOM BOM-BLR Del-HYD HYD-DEL

PLease share a table with following columns: Routes, Frequency of Air India, Air INdia express and Indigo


    """
    
    print(f"\n--- Running Postgres Verification with Gemini 2.0 Flash ---\nQuery: {query}")
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
