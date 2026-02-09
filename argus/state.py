from typing import Any, List, Optional

from pydantic import BaseModel, Field


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


class StepResult(BaseModel):
    """Result from executing a single analysis step."""
    step_id: int
    output: Any
    success: bool
    error: Optional[str] = None
    query_executed: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None


class AnalysisPlan(BaseModel):
    """The complete plan for analysis execution."""
    steps: List[AnalysisStep] = Field(description="List of analysis steps to perform")


class AnalysisContext(BaseModel):
    """Context passed through the analysis workflow."""
    user_query: str
    plan: List[AnalysisStep] = Field(default_factory=list)
    schema_summary: str = ""
    schema_dict: dict = Field(default_factory=dict)
    step_results: List[StepResult] = Field(default_factory=list)
    final_memo: str = ""
    debug_trace: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
