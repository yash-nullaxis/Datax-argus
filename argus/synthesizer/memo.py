"""
Executive memo synthesizer using Agno Agent.
"""
from agno.agent import Agent

from ..config import AgentConfig
from ..models import get_agno_model
from ..state import AnalysisContext, AnalysisStep


class Synthesizer:
    """Synthesizes analysis results into executive memos."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = get_agno_model(config.model)

    async def synthesize(self, context: AnalysisContext) -> str:
        """
        Generates a final answer (Executive Memo) based on all step results.
        """
        results_summary = ""
        plan = context.plan or []
        # Create a map for quick lookup
        step_map = {step.id: step for step in plan}

        for res in context.step_results:
            step = step_map.get(res.step_id)
            step_desc = step.description if step is not None else "Unknown Step"
            results_summary += f"Step {res.step_id}: {step_desc}\n"
            if res.query_executed:
                results_summary += f"Query: {res.query_executed}\n"
            if res.success:
                results_summary += f"Result: {res.output}\n\n"
            else:
                results_summary += f"Error: {res.error}\n\n"

        # Optionally append a trace appendix when verbose
        debug_lines = []
        if self.config.verbose and context.debug_trace:
            debug_lines.append("\n\nAppendix: Execution Trace\n")
            for entry in context.debug_trace:
                debug_lines.append(f"- {entry}")

        if debug_lines:
            results_with_trace = results_summary + "\n".join(debug_lines)
        else:
            results_with_trace = results_summary

        system_prompt = """You are an Executive Strategy Consultant. Write a final memo answering the user's original question based on the analysis results.

Structure:
1. Executive Summary (BLUF)
2. Key Findings (each finding should be self-contained and richly detailed, without phrases like "see Step 1/2/3"; repeat the relevant numbers and context inline)
3. Recommended Actions

Format: Markdown.
"""

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            markdown=True,
        )

        user_message = f"User Question: {context.user_query}\n\nAnalysis Results:\n{results_with_trace}"
        
        # Use async run
        response = await agent.arun(user_message)
        
        if hasattr(response, 'content') and response.content:
            return str(response.content)
        
        return str(response)

    def synthesize_sync(self, context: AnalysisContext) -> str:
        """
        Synchronous version of synthesize for non-async contexts.
        """
        results_summary = ""
        plan = context.plan or []
        step_map = {step.id: step for step in plan}

        for res in context.step_results:
            step = step_map.get(res.step_id)
            step_desc = step.description if step is not None else "Unknown Step"
            results_summary += f"Step {res.step_id}: {step_desc}\n"
            if res.query_executed:
                results_summary += f"Query: {res.query_executed}\n"
            if res.success:
                results_summary += f"Result: {res.output}\n\n"
            else:
                results_summary += f"Error: {res.error}\n\n"

        debug_lines = []
        if self.config.verbose and context.debug_trace:
            debug_lines.append("\n\nAppendix: Execution Trace\n")
            for entry in context.debug_trace:
                debug_lines.append(f"- {entry}")

        if debug_lines:
            results_with_trace = results_summary + "\n".join(debug_lines)
        else:
            results_with_trace = results_summary

        system_prompt = """You are an Executive Strategy Consultant. Write a final memo answering the user's original question based on the analysis results.

Structure:
1. Executive Summary (BLUF)
2. Key Findings (each finding should be self-contained and richly detailed, without phrases like "see Step 1/2/3"; repeat the relevant numbers and context inline)
3. Recommended Actions

Format: Markdown.
"""

        agent = Agent(
            model=self.model,
            instructions=system_prompt,
            markdown=True,
        )

        user_message = f"User Question: {context.user_query}\n\nAnalysis Results:\n{results_with_trace}"
        
        response = agent.run(user_message)
        
        if hasattr(response, 'content') and response.content:
            return str(response.content)
        
        return str(response)
