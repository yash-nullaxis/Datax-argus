"""
SQL Executor for running validated queries.
"""
from typing import Any, List, Dict

from ..db.connector import DBConnector


class SQLExecutor:
    """Executes SQL queries against a database connector."""
    
    def __init__(self, db: DBConnector):
        self.db = db
    
    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes the query and returns results.
        Raises exception on failure.
        """
        # In a real sandbox, we would have row limits, timeout logic here.
        try:
            return self.db.execute_query(query)
        except Exception as e:
            # Re-raise with clear message
            raise RuntimeError(f"Database Execution Error: {str(e)}") from e
