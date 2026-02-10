from enum import Enum
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """Classification of errors for intelligent replanning."""
    # Recoverable errors - worth replanning
    SCHEMA_ERROR = "schema_error"           # Table/column not found
    SYNTAX_ERROR = "syntax_error"           # SQL syntax issues
    TIMEOUT_ERROR = "timeout_error"         # Query took too long
    RELEVANCE_ERROR = "relevance_error"     # Results not relevant
    EMPTY_RESULT = "empty_result"           # No data returned
    
    # Fatal errors - replanning won't help
    CONNECTION_ERROR = "connection_error"   # DB unreachable
    PERMISSION_ERROR = "permission_error"   # Access denied
    SECURITY_VIOLATION = "security_violation"  # Blocked query
    
    # Unknown
    UNKNOWN = "unknown"


class FailedApproach(BaseModel):
    """Records a failed approach to prevent repeating mistakes."""
    step_description: str = Field(description="What was attempted")
    query_attempted: Optional[str] = Field(default=None, description="SQL query if applicable")
    error_category: ErrorCategory = Field(description="Type of failure")
    error_message: str = Field(description="Detailed error message")
    datasource: str = Field(description="Which datasource was targeted")
    tables_involved: List[str] = Field(default_factory=list, description="Tables that caused issues")
    columns_involved: List[str] = Field(default_factory=list, description="Columns that caused issues")
    
    def to_prompt_context(self) -> str:
        """Convert to a string for LLM context."""
        parts = [f"- FAILED: '{self.step_description}'"]
        if self.query_attempted:
            parts.append(f"  Query: {self.query_attempted[:200]}...")
        parts.append(f"  Error: {self.error_category.value} - {self.error_message}")
        if self.tables_involved:
            parts.append(f"  Avoid tables: {', '.join(self.tables_involved)}")
        if self.columns_involved:
            parts.append(f"  Avoid columns: {', '.join(self.columns_involved)}")
        return "\n".join(parts)


class ReplanRequest(BaseModel):
    """Request for the planner to generate a new plan after failures."""
    original_query: str = Field(description="The user's original question")
    failed_approaches: List[FailedApproach] = Field(default_factory=list)
    successful_results: Dict[int, Any] = Field(
        default_factory=dict, 
        description="Results from steps that succeeded (step_id -> result)"
    )
    remaining_goal: str = Field(
        default="", 
        description="What still needs to be accomplished"
    )
    replan_count: int = Field(default=0, description="How many times we've replanned")
    constraints: List[str] = Field(
        default_factory=list,
        description="Explicit constraints for the new plan"
    )


class AnalysisStep(BaseModel):
    """Represents a single step in the analysis plan."""
    id: int = Field(description="Unique identifier for the step")
    description: str = Field(description="Detailed description of the analysis step")
    tool: str = Field(description="Tool to use, e.g. 'sql'")
    datasource: str = Field(
        default="default", description="Data source name to target"
    )
    dependency: int = Field(
        default=-1, description="ID of a previous step this depends on, or -1 if none"
    )
    thought: str = Field(default="", description="Reasoning for this step")
    is_fallback: bool = Field(
        default=False, 
        description="True if this step is part of a fallback/replan"
    )


class StepResult(BaseModel):
    """Result from executing a single analysis step."""
    step_id: int
    output: Any
    success: bool
    error: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    query_executed: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    is_recoverable: bool = Field(
        default=True,
        description="Whether this failure might be recovered via replanning"
    )
    
    def to_failed_approach(self, step: "AnalysisStep") -> FailedApproach:
        """Convert a failed result to a FailedApproach for replanning context."""
        tables = []
        columns = []
        
        # Extract table/column names from error messages
        if self.error:
            error_lower = self.error.lower()
            if "table" in error_lower and "'" in self.error:
                # Try to extract table name from error like "Table 'foo' does not exist"
                import re
                table_match = re.findall(r"table\s*['\"]([^'\"]+)['\"]", error_lower)
                tables.extend(table_match)
            if "column" in error_lower and "'" in self.error:
                col_match = re.findall(r"column\s*['\"]([^'\"]+)['\"]", error_lower)
                columns.extend(col_match)
        
        return FailedApproach(
            step_description=step.description,
            query_attempted=self.query_executed,
            error_category=self.error_category or ErrorCategory.UNKNOWN,
            error_message=self.error or "Unknown error",
            datasource=step.datasource,
            tables_involved=tables,
            columns_involved=columns,
        )


class AnalysisPlan(BaseModel):
    """The complete plan for analysis execution."""
    steps: List[AnalysisStep] = Field(description="List of analysis steps to perform")
    reasoning: str = Field(default="", description="Overall reasoning for this plan")


class AnalysisContext(BaseModel):
    """Context passed through the analysis workflow."""
    user_query: str
    plan: List[AnalysisStep] = Field(default_factory=list)
    schema_summary: str = ""
    schema_dict: dict = Field(default_factory=dict)
    step_results: List[StepResult] = Field(default_factory=list)
    final_memo: str = ""
    debug_trace: List[str] = Field(default_factory=list)
    
    # Replanning state
    failed_approaches: List[FailedApproach] = Field(default_factory=list)
    replan_count: int = Field(default=0)
    successful_partial_results: Dict[int, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_recoverable_failures(self) -> List[StepResult]:
        """Get failed results that might be recoverable via replanning."""
        return [r for r in self.step_results if not r.success and r.is_recoverable]
    
    def has_fatal_error(self) -> bool:
        """Check if any fatal (non-recoverable) error occurred."""
        fatal_categories = {
            ErrorCategory.CONNECTION_ERROR,
            ErrorCategory.PERMISSION_ERROR,
            ErrorCategory.SECURITY_VIOLATION,
        }
        return any(
            r.error_category in fatal_categories 
            for r in self.step_results 
            if not r.success
        )
    
    def build_replan_request(self, remaining_goal: str = "") -> ReplanRequest:
        """Build a replan request from the current context."""
        return ReplanRequest(
            original_query=self.user_query,
            failed_approaches=self.failed_approaches,
            successful_results=self.successful_partial_results,
            remaining_goal=remaining_goal or self.user_query,
            replan_count=self.replan_count,
            constraints=[
                f"Avoid: {fa.tables_involved}" 
                for fa in self.failed_approaches 
                if fa.tables_involved
            ],
        )
