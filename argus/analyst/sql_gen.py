"""
SQL Generator using Agno Agent with structured output.
"""
import logging
from typing import Optional

from agno.agent import Agent
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import get_agno_model


logger = logging.getLogger(__name__)


class SQLQuery(BaseModel):
    """Structured SQL query output."""
    query: str = Field(description="Syntactically correct SQL query")
    explanation: str = Field(description="Explanation of what the query does")


class SQLSynthesizer:
    """Generates SQL queries using Agno Agent."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)

    def generate_sql(
        self,
        goal: str,
        schema_context: str,
        dialect: str = "duckdb",
        error_context: Optional[str] = None,
    ) -> str:
        """Generate SQL for a given goal."""
        
        # Dialect-aware guidance and examples
        dialect_hint = ""
        if dialect.lower().startswith("duck"):
            dialect_hint = (
                "Use DuckDB-compatible syntax. Example: "
                "SELECT date_trunc('day', ts) AS day, COUNT(*) FROM events "
                "GROUP BY day ORDER BY day;"
            )
        elif dialect.lower().startswith("post"):
            dialect_hint = (
                "Use PostgreSQL-compatible syntax. Example: "
                "SELECT date_trunc('day', ts) AS day, AVG(amount) FROM payments "
                "GROUP BY day ORDER BY day;"
            )

        base_prompt = (
            f"You are an expert {dialect} SQL writer. Write a single SELECT query to "
            "accomplish the user's goal.\n\n"
            "Rules:\n"
            "- Use the specific column names and table names from the schema.\n"
            "- Do not hallucinate columns or tables.\n"
            "- If the goal requires multiple steps efficiently expressible in one query "
            "(e.g. CTEs), you may use them.\n"
            "- Always include an explicit LIMIT if the goal does not require full-table scans.\n"
            "- If the schema clearly cannot answer the question, write a harmless query "
            "that returns zero rows (for example, WHERE 1 = 0) rather than inventing tables.\n"
            "- Return only the SELECT statement.\n"
        )

        if dialect_hint:
            base_prompt += "\n\nDialect guidance:\n" + dialect_hint + "\n"

        if error_context:
            base_prompt += (
                "\nCRITICAL: Your previous attempt failed with this error:\n"
                f"{error_context}\n"
                "Carefully fix your SQL logic or syntax so that the new query avoids this error."
            )

        agent = Agent(
            model=self.model,
            instructions=base_prompt,
            output_schema=SQLQuery,
            markdown=True,
        )

        user_message = f"Schema:\n{schema_context}\n\nGoal: {goal}"
        
        try:
            response = agent.run(user_message)
            
            # Extract query from structured response
            if hasattr(response, 'content') and response.content:
                result = response.content
                if isinstance(result, SQLQuery):
                    return result.query
                elif isinstance(result, dict):
                    return result.get("query") or result.get("sql") or ""
            
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, SQLQuery):
                    return response.data.query
                elif isinstance(response.data, dict):
                    return response.data.get("query") or response.data.get("sql") or ""
                    
            logger.warning(f"Could not extract SQL from response: {response}")
            return ""
            
        except Exception as e:
            logger.error(f"SQL Gen Error: {e}")
            return ""
