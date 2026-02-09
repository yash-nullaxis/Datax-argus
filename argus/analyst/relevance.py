"""
Relevance checker using Agno Agent with structured output.
"""
from agno.agent import Agent
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import get_agno_model


class RelevanceScore(BaseModel):
    """Structured relevance assessment."""
    is_relevant: bool = Field(
        description="Whether the result answers the specific part of the user query."
    )
    score: int = Field(description="Relevance score from 0 to 10.")
    reasoning: str = Field(description="Why this result is relevant or not.")


class RelevanceChecker:
    """Checks if SQL results are relevant to the analysis goal."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)

    def check_relevance(
        self, query: str, goal: str, result_summary: str
    ) -> RelevanceScore:
        """
        Checks if the SQL result is relevant to the step goal.
        """
        system_prompt = """
You are a Quality Assurance Analyst for data analysis.

Evaluate whether the SQL result actually answers the analytic goal.

Guidelines:
- Rate relevance from 0-10.
- If the result is empty, clearly off-topic, or aggregates the wrong metric
  (for example, counts rows when the goal asks for an average over time),
  the relevance should be low (0-3).
- If the result partially answers the goal but misses key constraints
  (wrong time grain, missing filters, wrong grouping), score it in the mid range (4-6).
- If the result directly and correctly answers the goal using appropriate filters,
  aggregations, and groupings, relevance should be high (7-10).
"""

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            output_schema=RelevanceScore,
            markdown=True,
        )

        user_message = f"Goal: {goal}\nQuery Executed: {query}\nResult Summary: {result_summary}"
        
        try:
            response = agent.run(user_message)
            
            if hasattr(response, 'content') and response.content:
                result = response.content
                if isinstance(result, RelevanceScore):
                    return result
                elif isinstance(result, dict):
                    return RelevanceScore(**result)
            
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, RelevanceScore):
                    return response.data
                elif isinstance(response.data, dict):
                    return RelevanceScore(**response.data)
            
            # Fallback: assume low relevance if parsing fails
            return RelevanceScore(
                is_relevant=False,
                score=0,
                reasoning="Failed to parse relevance response"
            )
            
        except Exception as e:
            return RelevanceScore(
                is_relevant=False,
                score=0,
                reasoning=f"Error checking relevance: {e}"
            )
