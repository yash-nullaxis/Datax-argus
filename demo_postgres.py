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
    Show me the total number of flights for each month in 2025.
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


async def full_result_demo():
    """Demonstrate getting full result without streaming."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    config = AgentConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-2.0-flash",
            api_key=SecretStr(api_key)
        ),
        db_connection_str=os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres"
        ),
        verbose=True,
        max_retry_per_step=3,
        min_relevance_score=5,
    )

    orchestrator = Orchestrator(config)

    query = "What is the total count of records in each table?"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    result = await orchestrator.run(query)
    
    print("\n--- PLAN ---")
    for step in result["plan"]:
        print(f"  [{step['id']}] {step['description']} (tool: {step['tool']})")
    
    print("\n--- RESULTS ---")
    for step_result in result["step_results"]:
        status = "✓" if step_result["success"] else "✗"
        print(f"  {status} Step {step_result['step_id']}")
        if step_result.get("query_executed"):
            print(f"    SQL: {step_result['query_executed'][:100]}...")
        if step_result.get("error"):
            print(f"    Error: {step_result['error']}")
    
    print("\n--- FINAL MEMO ---")
    print(result["final_memo"])


def sync_demo():
    """Demonstrate synchronous usage."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key found in environment.")
        return

    provider = "google" if os.getenv("GOOGLE_API_KEY") else "openai"
    model_name = "gemini-2.0-flash" if provider == "google" else "gpt-4-turbo"

    config = AgentConfig(
        model=ModelConfig(
            provider=provider,
            model_name=model_name,
            api_key=SecretStr(api_key)
        ),
        db_connection_str=os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:0h5UPFxhRWUFwdwE@localhost:5432/postgres"
        ),
        verbose=True,
    )

    orchestrator = Orchestrator(config)
    
    result = orchestrator.run_sync("What is the total count of records?")
    print(result["final_memo"])


if __name__ == "__main__":
    # Choose which demo to run:
    # asyncio.run(full_result_demo())  # Full result demo
    # sync_demo()                       # Sync demo
    
    asyncio.run(main())  # Streaming demo (default)
