"""
Orchestrator for the Argus Analysis Agent using Agno Workflow.

This replaces the LangGraph-based orchestration with Agno's workflow system.
The workflow follows a Planner -> Analyst -> Synthesizer pattern.
"""
import logging
from typing import Dict, List, Optional, Any

from agno.agent import Agent
from agno.workflow import Workflow

from .analyst.decomposer import QueryDecomposer
from .analyst.executor import SQLExecutor
from .analyst.relevance import RelevanceChecker
from .analyst.schema import SchemaInspector
from .analyst.sql_gen import SQLSynthesizer
from .config import AgentConfig
from .db.connector import DBConnector, DuckDBConnector, SqlAlchemyConnector
from .models import get_agno_model
from .safety import SQLValidator
from .state import AnalysisContext, AnalysisStep, StepResult
from .synthesizer.memo import Synthesizer


logger = logging.getLogger(__name__)


class PlannerAgent:
    """Agent responsible for planning analysis steps."""
    
    def __init__(self, config: AgentConfig, inspector: SchemaInspector, 
                 decomposer: QueryDecomposer, connectors: Dict[str, DBConnector],
                 default_source: str):
        self.config = config
        self.inspector = inspector
        self.decomposer = decomposer
        self.connectors = connectors
        self.default_source = default_source
    
    def plan(self, user_query: str) -> AnalysisContext:
        """Generate an analysis plan for the user query."""
        if self.config.verbose:
            logger.info("[Planner] Fetching Schema Summary...")
            
        schema_summary = self.inspector.get_summary(query=user_query)
        
        # Structured schema for validators / planner hints
        schema_dict = {}
        try:
            for name, conn in self.connectors.items():
                try:
                    schema_dict[name] = conn.get_schema_dict()
                except Exception:
                    schema_dict[name] = {}
        except Exception:
            schema_dict = {}
        
        if self.config.verbose:
            logger.info(f"[Planner] Schema Summary:\n{schema_summary}")
        
        # Decompose
        datasource_list = list(self.connectors.keys())
        steps = self.decomposer.decompose(
            user_query,
            context=schema_summary,
            datasources=self.config.datasource_hints or {
                name: "" for name in datasource_list
            },
        )

        if not steps:
            # Fallback to a single default SQL step so the agent remains usable.
            steps = [
                AnalysisStep(
                    id=1,
                    description=user_query,
                    tool="sql",
                    datasource=self.default_source,
                    dependency=-1,
                    thought="Fallback single-step plan because planner returned no steps.",
                )
            ]
        
        if self.config.verbose:
            logger.info(f"[Planner] Generated Plan: {len(steps)} steps")
            for s in steps:
                logger.info(f"  - [{s.id}] {s.description} (Source: {s.datasource})")
                
        debug_msgs = [
            "[Planner] Completed planning.",
            f"[Planner] Sources: {list(self.connectors.keys())}",
        ]

        return AnalysisContext(
            user_query=user_query,
            plan=steps,
            schema_summary=schema_summary,
            schema_dict=schema_dict,
            debug_trace=debug_msgs,
        )


class AnalystAgent:
    """Agent responsible for executing analysis steps."""
    
    def __init__(self, config: AgentConfig, sql_gen: SQLSynthesizer,
                 validator: SQLValidator, executor: SQLExecutor,
                 relevance_checker: RelevanceChecker,
                 connectors: Dict[str, DBConnector],
                 default_source: str):
        self.config = config
        self.sql_gen = sql_gen
        self.validator = validator
        self.executor = executor
        self.relevance_checker = relevance_checker
        self.connectors = connectors
        self.default_source = default_source

    def _infer_sqlglot_dialect(self, connector: DBConnector) -> str:
        """
        Infer a sqlglot-compatible dialect name for the active connector.
        """
        if isinstance(connector, DuckDBConnector):
            return "duckdb"

        engine = getattr(connector, "engine", None)
        dialect_name = None
        try:
            if engine is not None and hasattr(engine, "dialect"):
                dialect_name = getattr(engine.dialect, "name", None)
        except Exception:
            dialect_name = None

        if not dialect_name:
            return "duckdb"

        if dialect_name in ("postgresql", "postgres"):
            return "postgres"
        if dialect_name == "duckdb":
            return "duckdb"
        if dialect_name in ("sqlite",):
            return "sqlite"

        return str(dialect_name)

    def execute_step(self, step: AnalysisStep, context: AnalysisContext) -> StepResult:
        """Execute a single analysis step."""
        target_source = step.datasource or self.default_source
        if target_source not in self.connectors:
            target_source = self.default_source
             
        self.executor.db = self.connectors[target_source]
        
        debug_msgs = []
        
        if step.tool == "sql":
            schema_context = context.schema_summary
            
            # Local Retry Loop for Robustness
            last_error = None
            for attempt in range(self.config.max_retry_per_step):
                try:
                    if self.config.verbose:
                        logger.info(f"[Analyst] Step {step.id} (Attempt {attempt + 1})")

                    # Generate SQL
                    dialect = self._infer_sqlglot_dialect(self.connectors[target_source])
                    query = self.sql_gen.generate_sql(
                        step.description, 
                        schema_context,
                        dialect=dialect,
                        error_context=last_error,
                    )
                    
                    debug_msgs.append(f"[Analyst] Step {step.id} SQL: {query}")

                    # Validate
                    connector = self.connectors[target_source]
                    try:
                        connector_schema = (
                            connector.get_schema_dict()
                            if hasattr(connector, "get_schema_dict")
                            else {}
                        )
                    except Exception:
                        connector_schema = {}
                        
                    is_valid, error_msg = self.validator.validate(
                        query, dialect=dialect, schema_info=connector_schema
                    )
                    
                    if not is_valid:
                        last_error = f"Validation Error: {error_msg}"
                        debug_msgs.append(f"[Analyst] Validation failed: {error_msg}")
                        continue  # Retry

                    # Execute
                    result = self.executor.execute(query)
                    
                    # Relevance Check
                    rel = self.relevance_checker.check_relevance(
                        query, step.description, str(result)[:1000]
                    )
                    
                    if not rel.is_relevant or rel.score < self.config.min_relevance_score:
                        last_error = f"Low relevance: {rel.reasoning}"
                        debug_msgs.append(f"[Analyst] Low relevance: {last_error}")
                        continue  # Retry
                        
                    # Success
                    row_count = len(result) if isinstance(result, list) else None
                    columns = list(result[0].keys()) if row_count and row_count > 0 else None
                    
                    return StepResult(
                        step_id=step.id,
                        output=result,
                        success=True,
                        query_executed=query,
                        row_count=row_count,
                        columns=columns
                    )

                except Exception as e:
                    last_error = str(e)
                    debug_msgs.append(f"[Analyst] Error attempt {attempt+1}: {e}")
            
            # Specific failure after retries
            return StepResult(
                step_id=step.id,
                output=None,
                success=False,
                error=last_error or "Max retries exceeded"
            )
        else:
            return StepResult(
                step_id=step.id,
                output="Skipped non-SQL step",
                success=False,
                error="Unsupported tool"
            )

    def execute_all(self, context: AnalysisContext) -> AnalysisContext:
        """Execute all steps in the plan, respecting dependencies."""
        plan = context.plan
        results = []
        completed_ids = set()
        failed_ids = set()
        
        # Process steps in dependency order
        max_iterations = len(plan) * 2  # Safety limit
        iterations = 0
        
        while len(completed_ids) + len(failed_ids) < len(plan) and iterations < max_iterations:
            iterations += 1
            made_progress = False
            
            for step in plan:
                if step.id in completed_ids or step.id in failed_ids:
                    continue
                
                # Check dependencies
                if step.dependency == -1 or step.dependency in completed_ids:
                    # Execute this step
                    result = self.execute_step(step, context)
                    results.append(result)
                    
                    if result.success:
                        completed_ids.add(step.id)
                    else:
                        failed_ids.add(step.id)
                    
                    made_progress = True
                elif step.dependency in failed_ids:
                    # Dependency failed, skip this step
                    results.append(StepResult(
                        step_id=step.id,
                        output=None,
                        success=False,
                        error=f"Dependency step {step.dependency} failed"
                    ))
                    failed_ids.add(step.id)
                    made_progress = True
            
            if not made_progress:
                # No progress made, break to avoid infinite loop
                break
        
        context.step_results = results
        return context


class Orchestrator:
    """
    Main orchestrator for the Argus Analysis Agent using Agno.
    
    This orchestrates the Planner -> Analyst -> Synthesizer workflow.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        connector: Optional[DBConnector] = None,
        connectors: Optional[Dict[str, DBConnector]] = None,
    ):
        self.config = config
        
        # Initialize Connectors
        if connectors:
            self.connectors = connectors
            self.default_source = list(connectors.keys())[0]
            rag_store = connectors.get('metadata_store')
            self.inspector = SchemaInspector(self.connectors, vector_store=rag_store) 
            self.executor = SQLExecutor(self.connectors[self.default_source])
        elif connector:
            self.connectors = {"default": connector}
            self.default_source = "default"
            self.inspector = SchemaInspector(connector)
            self.executor = SQLExecutor(connector)
        else:
            default_conn = SqlAlchemyConnector(config.db_connection_str)
            self.connectors = {"default": default_conn}
            self.default_source = "default"
            self.inspector = SchemaInspector(default_conn)
            self.executor = SQLExecutor(default_conn)

        # Initialize Services
        self.decomposer = QueryDecomposer(config)
        self.sql_gen = SQLSynthesizer(config)
        self.validator = SQLValidator()
        self.relevance_checker = RelevanceChecker(config)
        self.synthesizer = Synthesizer(config)
        
        # Initialize Agents
        self.planner = PlannerAgent(
            config, self.inspector, self.decomposer, 
            self.connectors, self.default_source
        )
        self.analyst = AnalystAgent(
            config, self.sql_gen, self.validator, self.executor,
            self.relevance_checker, self.connectors, self.default_source
        )

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Main entry point to run the agent.
        """
        if self.config.verbose:
            logger.info("[Run] Starting analysis.")
        
        # Step 1: Planning
        context = self.planner.plan(query)
        
        # Step 2: Analysis
        context = self.analyst.execute_all(context)
        
        # Step 3: Synthesis
        final_memo = await self.synthesizer.synthesize(context)
        context.final_memo = final_memo
        
        return {
            "user_query": context.user_query,
            "plan": [step.model_dump() for step in context.plan],
            "step_results": [result.model_dump() for result in context.step_results],
            "final_memo": context.final_memo,
            "debug_trace": context.debug_trace,
        }

    def run_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous entry point to run the agent.
        """
        if self.config.verbose:
            logger.info("[Run] Starting analysis.")
        
        # Step 1: Planning
        context = self.planner.plan(query)
        
        # Step 2: Analysis
        context = self.analyst.execute_all(context)
        
        # Step 3: Synthesis
        final_memo = self.synthesizer.synthesize_sync(context)
        context.final_memo = final_memo
        
        return {
            "user_query": context.user_query,
            "plan": [step.model_dump() for step in context.plan],
            "step_results": [result.model_dump() for result in context.step_results],
            "final_memo": context.final_memo,
            "debug_trace": context.debug_trace,
        }

    async def stream_run(self, query: str):
        """
        Runs the agent and streams the final memo generation.
        """
        if self.config.verbose:
            logger.info("[Run] Starting streaming analysis.")
        
        # Step 1: Planning
        context = self.planner.plan(query)
        
        # Step 2: Analysis
        context = self.analyst.execute_all(context)
        
        # Step 3: Streaming Synthesis
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

        system_prompt = """You are an Executive Strategy Consultant. Write a final memo answering the user's original question based on the analysis results.

Structure:
1. Executive Summary (BLUF)
2. Key Findings (each finding should be self-contained and richly detailed, without phrases like "see Step 1/2/3"; repeat the relevant numbers and context inline)
3. Recommended Actions

Format: Markdown.
"""

        model = get_agno_model(self.config.model)
        agent = Agent(
            model=model,
            instructions=system_prompt,
            markdown=True,
        )

        user_message = f"User Question: {context.user_query}\n\nAnalysis Results:\n{results_summary}"
        
        # Stream the response using Agno's streaming API
        # When stream=True, arun returns an async generator of RunOutputEvent objects
        response_stream = agent.arun(user_message, stream=True)
        
        # Iterate over the streaming response
        async for event in response_stream:
            # Handle different types of streaming events from Agno
            # RunContent events contain incremental content deltas
            if hasattr(event, 'content_delta') and event.content_delta:
                yield str(event.content_delta)
            elif hasattr(event, 'content') and event.content:
                # Full content (may come at the end)
                yield str(event.content)
            elif hasattr(event, 'delta') and event.delta:
                yield str(event.delta)
            elif isinstance(event, str):
                yield event
