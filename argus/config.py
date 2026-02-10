from typing import Dict, Optional, Any, Literal

from pydantic import BaseModel, Field, SecretStr


class ModelConfig(BaseModel):
    """
    Configuration for the underlying chat model.

    This stays provider-agnostic but exposes a few knobs that map cleanly onto
    the supported Agno model wrappers.
    """

    provider: Literal["openai", "anthropic", "google"] = "openai"
    model_name: str = "gpt-4-turbo"
    api_key: SecretStr
    api_base: Optional[str] = None

    # Generation controls
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    timeout_s: Optional[float] = None

    # Escape hatch for provider-specific options
    extra: Dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """
    Top-level configuration for the Argus agent.
    """

    model: ModelConfig
    db_connection_str: str

    # Global graph limits
    max_steps: int = 15
    recursion_limit: int = 25

    # Execution controls
    max_rows_per_query: Optional[int] = None
    max_retry_per_step: int = 3
    
    # Replanning controls
    max_replan_attempts: int = 2  # How many times to replan after failures
    replan_on_partial_failure: bool = True  # Replan even if some steps succeeded
    replan_min_failure_ratio: float = 0.3  # Replan if >= 30% of steps failed

    # Logging / tracing
    verbose: bool = False
    log_sql: bool = False
    log_results: bool = False

    # Planning hints
    datasource_hints: Dict[str, str] = Field(
        default_factory=dict,
        description="Free-form hints about what each datasource is best for.",
    )

    # Quality thresholds
    min_relevance_score: int = 5
