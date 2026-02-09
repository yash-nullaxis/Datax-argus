# Argus SDK Documentation (Agno Implementation)

This document provides detailed technical documentation for the Argus SDK, an autonomous business intelligence agent built with the Agno framework.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Configuration](#configuration)
4. [Database Connectors](#database-connectors)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Advanced Usage](#advanced-usage)

---

## Architecture Overview

Argus follows a **Planner-Analyst-Synthesizer** pattern, orchestrated using Agno agents instead of LangGraph's state machine.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                              │
│                                                                   │
│   ┌───────────┐    ┌───────────┐    ┌─────────────────┐         │
│   │  Planner  │ -> │  Analyst  │ -> │   Synthesizer   │         │
│   │   Agent   │    │   Agent   │    │     Agent       │         │
│   └───────────┘    └───────────┘    └─────────────────┘         │
│         │                │                    │                   │
│         v                v                    v                   │
│   Query Decomposer  SQL Generator    Executive Memo Generator    │
│   Schema Inspector  SQL Executor                                  │
│                     Relevance Checker                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Differences from LangGraph Version

| Aspect | LangGraph Version | Agno Version |
|--------|------------------|--------------|
| Orchestration | StateGraph with nodes/edges | Direct agent coordination |
| LLM Interface | LangChain chat models | Agno models (OpenAI, Claude, Gemini) |
| State Management | TypedDict with reducers | Pydantic BaseModel context |
| Structured Output | JsonOutputParser | Agent output_schema |
| Streaming | astream_events | arun_stream |

---

## Core Components

### Orchestrator

The `Orchestrator` class is the main entry point for running analyses.

```python
from argus.orchestrator import Orchestrator
from argus.config import AgentConfig, ModelConfig

config = AgentConfig(
    model=ModelConfig(
        provider="google",
        model_name="gemini-2.0-flash",
        api_key="..."
    ),
    db_connection_str="postgresql://..."
)

orchestrator = Orchestrator(config)
```

#### Methods

| Method | Description |
|--------|-------------|
| `run(query: str)` | Async execution, returns full result dict |
| `run_sync(query: str)` | Synchronous execution |
| `stream_run(query: str)` | Async generator yielding memo tokens |

### PlannerAgent

Responsible for decomposing user queries into analysis steps.

**Input**: Natural language query + schema context  
**Output**: List of `AnalysisStep` objects

### AnalystAgent

Executes analysis steps with retry logic and relevance checking.

**Features**:
- SQL generation with dialect awareness
- Syntax and schema validation
- Result relevance scoring
- Auto-retry on failures

### Synthesizer

Generates executive memos from analysis results using an Agno agent.

---

## Configuration

### ModelConfig

```python
class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic", "google"] = "openai"
    model_name: str = "gpt-4-turbo"
    api_key: SecretStr
    api_base: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    timeout_s: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
```

### AgentConfig

```python
class AgentConfig(BaseModel):
    model: ModelConfig
    db_connection_str: str
    max_steps: int = 15
    recursion_limit: int = 25
    max_rows_per_query: Optional[int] = None
    max_retry_per_step: int = 3
    verbose: bool = False
    log_sql: bool = False
    log_results: bool = False
    datasource_hints: Dict[str, str] = Field(default_factory=dict)
    min_relevance_score: int = 5
```

---

## Database Connectors

### SqlAlchemyConnector

For PostgreSQL, MySQL, SQLite, and other SQLAlchemy-supported databases.

```python
from argus.db.connector import SqlAlchemyConnector

connector = SqlAlchemyConnector(
    "postgresql://user:pass@localhost:5432/db"
)
```

### DuckDBConnector

For local file analysis (CSV, Parquet, JSON, Excel).

```python
from argus.db.connector import DuckDBConnector

connector = DuckDBConnector(
    files=["data.csv", "logs.parquet"]
)
```

### MongoDBConnector

For MongoDB document stores.

```python
from argus.db.advanced_connectors import MongoDBConnector

connector = MongoDBConnector(
    connection_str="mongodb://localhost:27017",
    database="mydb"
)
```

### ChromaDBConnector

For vector search and RAG contexts.

```python
from argus.db.advanced_connectors import ChromaDBConnector

connector = ChromaDBConnector(collection_name="docs")
```

### Multi-Source Setup

```python
orchestrator = Orchestrator(
    config,
    connectors={
        "postgres": SqlAlchemyConnector("postgresql://..."),
        "files": DuckDBConnector(files=["data.csv"]),
        "metadata_store": ChromaDBConnector()  # Used for RAG
    }
)
```

---

## Analysis Pipeline

### Step 1: Query Decomposition

The QueryDecomposer uses an Agno agent with structured output:

```python
agent = Agent(
    model=model,
    instructions=system_prompt,
    output_schema=AnalysisPlan,
)
```

### Step 2: SQL Generation

The SQLSynthesizer generates dialect-aware SQL:

- **DuckDB**: Optimized for analytics
- **PostgreSQL**: Standard SQL with extensions
- **SQLite**: Lightweight SQL

### Step 3: Validation

SQLValidator checks:
- Syntax correctness (via sqlglot)
- Read-only safety (blocks mutations)
- Schema existence (tables and columns)
- Complexity limits (max tables/joins)

### Step 4: Execution

SQLExecutor runs validated queries and returns results as list of dicts.

### Step 5: Relevance Check

RelevanceChecker scores results 0-10:
- **0-3**: Off-topic or wrong aggregation
- **4-6**: Partially correct
- **7-10**: Accurate answer

### Step 6: Synthesis

The Synthesizer generates a markdown executive memo:

1. Executive Summary (BLUF)
2. Key Findings
3. Recommended Actions

---

## Advanced Usage

### Custom Providers

```python
from agno.models.openai import OpenAIChat

# Azure OpenAI
model = OpenAIChat(
    id="gpt-4",
    api_key="...",
    base_url="https://your-resource.openai.azure.com/",
    default_headers={"api-key": "..."}
)
```

### RAG-Enhanced Schema Context

With a ChromaDBConnector as `metadata_store`:

```python
orchestrator = Orchestrator(
    config,
    connectors={
        "main": SqlAlchemyConnector("postgresql://..."),
        "metadata_store": ChromaDBConnector()
    }
)
```

The SchemaInspector will:
1. Index table schemas into ChromaDB
2. Retrieve relevant tables for each query
3. Reduce context size for large databases

### Debugging

Enable verbose mode:

```python
config = AgentConfig(
    verbose=True,
    log_sql=True,
    log_results=True,
    ...
)
```

Check debug trace in results:

```python
result = await orchestrator.run(query)
for trace in result["debug_trace"]:
    print(trace)
```

---

## Model Support

### OpenAI

```python
ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo",  # or gpt-4o, gpt-3.5-turbo
    api_key="sk-..."
)
```

### Anthropic (Claude)

```python
ModelConfig(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    api_key="..."
)
```

### Google (Gemini)

```python
ModelConfig(
    provider="google",
    model_name="gemini-2.0-flash",
    api_key="..."
)
```

---

## Error Handling

### Common Errors

1. **SafetyError**: Query contains mutation (UPDATE, DELETE, etc.)
2. **Schema Error**: Table or column doesn't exist
3. **Validation Error**: SQL syntax error
4. **Relevance Error**: Results don't match the goal

### Retry Logic

Each step retries up to `max_retry_per_step` times with:
- Error context fed back to LLM
- Progressive refinement

---

## Performance Tips

1. **Limit schema profiling**: Large databases can slow startup
2. **Use RAG**: For 50+ tables, use ChromaDB for schema context
3. **Tune relevance threshold**: Lower `min_relevance_score` for exploratory queries
4. **Batch queries**: Combine related questions for efficiency

---

## API Reference

### State Models

```python
class AnalysisStep(BaseModel):
    id: int
    description: str
    tool: str  # "sql" or "python"
    datasource: str = "default"
    dependency: int = -1  # -1 = no dependency
    thought: str = ""

class StepResult(BaseModel):
    step_id: int
    output: Any
    success: bool
    error: Optional[str] = None
    query_executed: Optional[str] = None
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None

class AnalysisContext(BaseModel):
    user_query: str
    plan: List[AnalysisStep]
    schema_summary: str
    schema_dict: dict
    step_results: List[StepResult]
    final_memo: str
    debug_trace: List[str]
```

### Result Structure

```python
{
    "user_query": "...",
    "plan": [...],
    "step_results": [...],
    "final_memo": "...",
    "debug_trace": [...]
}
```
