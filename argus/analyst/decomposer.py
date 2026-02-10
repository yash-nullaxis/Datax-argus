"""
Query decomposer using Agno Agent for step planning.
Supports initial planning and dynamic replanning after failures.
"""
import logging
from typing import Dict, List, Optional

from agno.agent import Agent
from pydantic import BaseModel, Field

from ..config import AgentConfig
from ..models import get_agno_model
from ..state import AnalysisStep, AnalysisPlan, ReplanRequest, FailedApproach


logger = logging.getLogger(__name__)


INITIAL_PLAN_SYSTEM_PROMPT = """\
You are a senior data analyst. Your goal is to create a step-by-step \
execution plan to answer the user's business question.

Capabilities:
- You have access to multiple data sources (SQL, Vector, Files).
- You can execute SQL queries.
- You can perform final synthesis.

Available Tools:
- sql: Generate and execute a SQL query.
- python: (Not yet implemented, generally avoid unless necessary).

Rules:
- Break down complex 'compare' questions into separate data retrieval steps.
- Prefer a small number of high-quality steps (typically 3-7).
- Ensure steps are logical and sequential. Assign unique IDs to steps.
- If a step requires the result of a previous step (e.g. 'Compare X and Y' \
needs 'Get X'), set 'dependency' to the ID of that previous step.
- Use the provided schema context to infer feasibility (e.g. if table exists).
- IMPORTANT: For each step, choose the most appropriate 'datasource' \
from the list of available data sources. If unsure, use 'default'.
"""

REPLAN_SYSTEM_PROMPT = """\
You are a senior data analyst performing RECOVERY PLANNING after some analysis steps failed.

Your task is to create a NEW plan that avoids the mistakes from previous attempts.

CRITICAL RULES:
1. DO NOT repeat approaches that already failed. Use alternative tables, columns, or strategies.
2. If a table doesn't exist, find an alternative table with similar data.
3. If a column doesn't exist, check if the data might be in a different column or derived.
4. If a query timed out, break it into smaller, simpler queries.
5. Learn from the errors - they contain valuable information about what WON'T work.
6. You may reuse successful results from previous steps - don't re-query data you already have.

Available Tools:
- sql: Generate and execute a SQL query.
- python: (Not yet implemented, generally avoid unless necessary).

Output a plan that will SUCCEED where the previous plan failed.
"""


class QueryDecomposer:
    """Decomposes user queries into analysis steps using Agno."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)
    
    def _build_datasource_description(self, datasources: Optional[Dict[str, str]]) -> str:
        """Build a description of available datasources."""
        if not datasources:
            return ""
        
        parts = []
        for name, hint in datasources.items():
            label = f"- {name}"
            if hint:
                label += f": {hint}"
            parts.append(label)
        return "\nYou have access to the following data sources:\n" + "\n".join(parts)
    
    def _build_failure_context(self, failed_approaches: List[FailedApproach]) -> str:
        """Build context about failed approaches to avoid repeating mistakes."""
        if not failed_approaches:
            return ""
        
        lines = [
            "\n⚠️ PREVIOUS FAILED APPROACHES (DO NOT REPEAT THESE):",
            "=" * 50
        ]
        for fa in failed_approaches:
            lines.append(fa.to_prompt_context())
            lines.append("")
        lines.append("=" * 50)
        lines.append("Create a NEW plan that avoids these failures.\n")
        
        return "\n".join(lines)
    
    def _build_success_context(self, successful_results: Dict[int, any]) -> str:
        """Build context about successful results that can be reused."""
        if not successful_results:
            return ""
        
        lines = [
            "\n✅ ALREADY RETRIEVED DATA (can be referenced, no need to re-query):",
        ]
        for step_id, result in successful_results.items():
            # Truncate large results
            result_str = str(result)[:500]
            lines.append(f"  Step {step_id} result: {result_str}")
        lines.append("")
        
        return "\n".join(lines)

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
        ds_description = self._build_datasource_description(datasources)
        
        system_prompt = INITIAL_PLAN_SYSTEM_PROMPT
        if ds_description:
            system_prompt += "\n" + ds_description + "\n"

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            output_schema=AnalysisPlan,
            markdown=True,
        )

        user_message = f"Schema Context:\n{context}\n\nUser Question: {query}"
        
        return self._execute_planning(agent, user_message)
    
    def replan(
        self,
        request: ReplanRequest,
        schema_context: str = "",
        datasources: Optional[Dict[str, str]] = None,
    ) -> List[AnalysisStep]:
        """
        Generate a new plan after failures, avoiding previous mistakes.
        
        This is the core of dynamic replanning - it takes context about
        what failed and generates an alternative approach.
        """
        ds_description = self._build_datasource_description(datasources)
        failure_context = self._build_failure_context(request.failed_approaches)
        success_context = self._build_success_context(request.successful_results)
        
        system_prompt = REPLAN_SYSTEM_PROMPT
        if ds_description:
            system_prompt += "\n" + ds_description + "\n"

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            output_schema=AnalysisPlan,
            markdown=True,
        )

        # Build comprehensive user message with all context
        user_message_parts = [
            f"Original User Question: {request.original_query}",
            "",
            f"Remaining Goal: {request.remaining_goal}",
            "",
            f"Replan Attempt: {request.replan_count + 1}",
        ]
        
        if request.constraints:
            user_message_parts.append("\nConstraints:")
            for c in request.constraints:
                user_message_parts.append(f"  - {c}")
        
        user_message_parts.append(f"\nSchema Context:\n{schema_context}")
        user_message_parts.append(failure_context)
        user_message_parts.append(success_context)
        
        user_message = "\n".join(user_message_parts)
        
        if self.config.verbose:
            logger.info(f"[Decomposer] Replanning (attempt {request.replan_count + 1})")
            logger.info(f"[Decomposer] Failed approaches: {len(request.failed_approaches)}")
        
        steps = self._execute_planning(agent, user_message)
        
        # Mark all replan steps as fallback
        for step in steps:
            step.is_fallback = True
        
        return steps
    
    def _execute_planning(self, agent: Agent, user_message: str) -> List[AnalysisStep]:
        """Execute the planning agent and parse the response."""
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
            logger.error(f"Planning Error: {e}")
            return []
