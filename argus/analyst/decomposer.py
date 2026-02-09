"""
Query decomposer using Agno Agent for step planning.
"""
import logging
from typing import Dict, List, Optional

from agno.agent import Agent
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import get_agno_model
from ..state import AnalysisStep, AnalysisPlan


logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Decomposes user queries into analysis steps using Agno."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)
    
    def decompose(
        self,
        query: str,
        context: str = "",
        datasources: Optional[Dict[str, str]] = None,
    ) -> List[AnalysisStep]:
        """
        Decomposes a user query into specific analysis steps.
        Now includes 'context' (schema info) to help the planner make better decisions.
        """
        ds_description = ""
        if datasources:
            parts = []
            for name, hint in datasources.items():
                label = f"- {name}"
                if hint:
                    label += f": {hint}"
                parts.append(label)
            ds_description = "\nYou have access to the following data sources:\n" + "\n".join(parts)
        
        system_prompt = (
            "You are a senior data analyst. Your goal is to create a step-by-step "
            "execution plan to answer the user's business question.\n\n"
            "Capabilities:\n"
            "- You have access to multiple data sources (SQL, Vector, Files).\n"
            "- You can execute SQL queries.\n"
            "- You can perform final synthesis.\n\n"
            "Available Tools:\n"
            "- sql: Generate and execute a SQL query.\n"
            "- python: (Not yet implemented, generally avoid unless necessary).\n\n"
            "Rules:\n"
            "- Break down complex 'compare' questions into separate data retrieval steps.\n"
            "- Prefer a small number of high-quality steps (typically 3-7).\n"
            "- Ensure steps are logical and sequential. Assign unique IDs to steps.\n"
            "- If a step requires the result of a previous step (e.g. 'Compare X and Y' "
            "needs 'Get X'), set 'dependency' to the ID of that previous step.\n"
            "- Use the provided schema context to infer feasibility (e.g. if table exists).\n"
            "- IMPORTANT: For each step, choose the most appropriate 'datasource' "
            "from the list of available data sources. If unsure, use 'default'.\n\n"
        )

        if ds_description:
            system_prompt += "\n" + ds_description + "\n"

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            output_schema=AnalysisPlan,
            markdown=True,
        )

        user_message = f"Schema Context:\n{context}\n\nUser Question: {query}"
        
        try:
            response = agent.run(user_message)
            
            # Extract the structured output
            if hasattr(response, 'content') and response.content:
                plan = response.content
                if isinstance(plan, AnalysisPlan):
                    return plan.steps
                elif isinstance(plan, dict):
                    parsed_plan = AnalysisPlan(**plan)
                    return parsed_plan.steps
            
            # Fallback: try to parse from response
            if hasattr(response, 'data') and response.data:
                if isinstance(response.data, AnalysisPlan):
                    return response.data.steps
                elif isinstance(response.data, dict):
                    parsed_plan = AnalysisPlan(**response.data)
                    return parsed_plan.steps
                    
            logger.warning(f"Could not parse plan from response: {response}")
            return []
            
        except Exception as e:
            logger.error(f"Decomposition Error: {e}")
            return []
