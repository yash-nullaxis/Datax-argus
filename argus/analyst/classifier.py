"""
Query classifier using Agno Agent to determine query complexity.
"""
from agno.agent import Agent
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import get_agno_model


class ComplexityResult(BaseModel):
    """Result of query complexity classification."""
    is_complex: bool = Field(
        description="True if the query is complex and requires decomposition to multiple steps, "
                    "False if it is simple and can be answered with a single SQL query."
    )
    reasoning: str = Field(description="The reasoning behind the complexity classification.")


class QueryClassifier:
    """Classifies user queries as simple or complex using Agno Agent."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)

    def classify(self, query: str) -> ComplexityResult:
        """Classify a query as simple or complex."""
        system_prompt = (
            "You are a smart assistant that classifies user queries for a data analysis agent.\n"
            "Your job is to determine if a query is 'Simple' (can be answered with a single direct SQL query) "
            "or 'Complex' (requires decomposition, multiple steps, finding multiple data points, or data profiling/reasoning).\n\n"
            "Examples of Simple Queries:\n"
            "- 'How many users are there?'\n"
            "- 'Show me the top 5 products by sales.'\n"
            "- 'List all flights in 2025.'\n"
            "- 'What is the average price of items?'\n\n"
            "Examples of Complex Queries:\n"
            "- 'Compare the sales performance of region A vs region B and explain the trend.'\n"
            "- 'Find anomalies in user signups and correlated events.'\n"
            "- 'Why did revenue drop last month?'\n"
            "- 'Get me the users who bought X and then Y within 3 days.'\n"
            "- 'Analyze the retention rate by cohort.'\n"
        )
        
        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            output_schema=ComplexityResult,
            markdown=True,
        )
        
        try:
            response = agent.run(f"Query: {query}")
            
            if hasattr(response, 'content') and response.content:
                result = response.content
                if isinstance(result, ComplexityResult):
                    return result
                elif isinstance(result, dict):
                    return ComplexityResult(**result)
            
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, ComplexityResult):
                    return response.data
                elif isinstance(response.data, dict):
                    return ComplexityResult(**response.data)
                    
            # Fallback to complex (safer to decompose)
            return ComplexityResult(
                is_complex=True, 
                reasoning="Could not parse classification response, defaulting to complex."
            )
            
        except Exception as e:
            # Fallback to true (complex) to be safe if classification fails
            return ComplexityResult(
                is_complex=True, 
                reasoning=f"Error during classification: {e}"
            )
