"""
Demo script for Argus SDK using a local SQLite database.
This allows you to test the agent without any external database setup.
"""
import asyncio
import os
import argparse
import sqlite3
import pandas as pd
from dotenv import load_dotenv

# Load .env to get API key and other config
load_dotenv()

from argus import Orchestrator, AgentConfig, ModelConfig, SqlAlchemyConnector
from pydantic import SecretStr


def setup_sample_db(db_path: str):
    """Creates a sample SQLite database with some data."""
    conn = sqlite3.connect(db_path)
    
    # Create sample tables
    # 1. Users table
    users_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'username': ['alice', 'bob', 'charlie', 'david', 'eve'],
        'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],
        'status': ['active', 'active', 'inactive', 'active', 'pending']
    })
    users_df.to_sql('users', conn, if_exists='replace', index=False)
    
    # 2. Orders table
    orders_df = pd.DataFrame({
        'order_id': [101, 102, 103, 104, 105, 106],
        'user_id': [1, 1, 2, 3, 4, 1],
        'amount': [250.0, 150.5, 300.2, 50.0, 450.7, 120.0],
        'order_date': ['2023-06-01', '2023-06-05', '2023-06-10', '2023-06-15', '2023-06-20', '2023-06-25']
    })
    orders_df.to_sql('orders', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Sample database created at {db_path}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable aggressive logging")
    args = parser.parse_args()

    db_path = "sample_data.db"
    setup_sample_db(db_path)
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in environment.")
        return

    # Determine provider based on which key is available
    provider = "google" if os.getenv("GOOGLE_API_KEY") else "openai"
    model_name = "gemini-2.0-flash" if provider == "google" else "gpt-4-turbo"
    
    config = AgentConfig(
        model=ModelConfig(
            provider=provider,
            model_name=model_name,
            api_key=SecretStr(api_key)
        ),
        db_connection_str=f"sqlite:///{db_path}",
        verbose=args.verbose,
        log_sql=True,
    )

    # Use a direct SQLAlchemy connector for SQLite
    connector = SqlAlchemyConnector(config.db_connection_str)
    orchestrator = Orchestrator(config, connector=connector)

    query = "What is the total order amount for each user? Show their usernames."
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    try:
        print("\n--- Final Memo (Streaming) ---")
        full_memo = ""
        async for chunk in orchestrator.stream_run(query):
            print(chunk, end="", flush=True)
            full_memo += chunk
            
        print("\n\n[Done]")
        
    except Exception as e:
        print(f"\n[Error] Agent execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


async def full_result_demo():
    """Demonstrate getting full result without streaming."""
    db_path = "sample_data.db"
    setup_sample_db(db_path)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No API key found.")
        return

    provider = "google" if os.getenv("GOOGLE_API_KEY") else "openai"
    model_name = "gemini-2.0-flash" if provider == "google" else "gpt-4-turbo"

    config = AgentConfig(
        model=ModelConfig(
            provider=provider,
            model_name=model_name,
            api_key=SecretStr(api_key)
        ),
        db_connection_str=f"sqlite:///{db_path}",
        verbose=True,
    )

    connector = SqlAlchemyConnector(config.db_connection_str)
    orchestrator = Orchestrator(config, connector=connector)

    query = "What is the total order amount for each user? Show their usernames."
    
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


if __name__ == "__main__":
    # Choose which demo to run:
    # asyncio.run(full_result_demo())  # Full result demo
    
    asyncio.run(main())  # Streaming demo (default)
