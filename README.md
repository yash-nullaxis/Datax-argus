# Argus SDK (Agno Implementation)

Autonomous Business Intelligence Agent for multi-source data connectivity, analysis, and synthesis. This implementation uses the **Agno** framework instead of LangGraph, providing a more streamlined agent-based architecture.

## 🚀 Key Features

* **Multi-Source Fusion**: Connect to multiple databases (SQL, NoSQL, Vector DB) simultaneously.
* **LLM-Driven Planning**: Automatically decomposes complex user queries into step-by-step analysis plans using Agno agents.
* **Self-Healing & Auto-Repair**: Automatically detects SQL errors or irrelevant results and attempts to fix the query.
* **Schema-Aware**: Uses schema inspection (and optional RAG context) to generate accurate SQL.
* **Safety & Validation**: Validates generated SQL using `sqlglot` before execution to prevent errors and ensure safety.
* **Agno-Powered**: Uses Agno's Agent and Workflow primitives for orchestration.

---

## 🏗️ Architecture

Argus follows a **Planner-Analyst-Synthesizer** architecture using Agno agents.

### Core Components

1. **Orchestrator (`argus.orchestrator.Orchestrator`)**:
   * The central controller that manages the workflow.
   * Coordinates the Planner, Analyst, and Synthesizer agents.
   * Manages the analysis context and state.

2. **Planner (`argus.analyst.decomposer.QueryDecomposer`)**:
   * Uses an Agno Agent to analyze the user's natural language request.
   * Generates a list of analysis steps with structured output.

3. **Analyst (`argus.analyst.*`)**:
   * Executes the plan step-by-step.
   * Contains components for SQL synthesis, validation, execution, and relevance checking.
   * All components use Agno agents for LLM interactions.

4. **Synthesizer (`argus.synthesizer.memo.Synthesizer`)**:
   * Aggregates results and generates a final executive memo using an Agno agent.

---

## 🛠️ Installation

```bash
pip install .
```

Or with poetry:

```bash
poetry install
```

---

## ⚙️ Configuration

Argus is configured via the `AgentConfig` object.

### `AgentConfig` Summary

| Field | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `model` | `ModelConfig` | LLM configuration (provider, model_name, api_key). | - |
| `db_connection_str` | `str` | Default database connection string. | - |
| `max_steps` | `int` | Maximum analysis steps (limit). | 15 |
| `max_retry_per_step` | `int` | Retries for SQL correction per step. | 3 |
| `verbose` | `bool` | Enable detailed logging. | `False` |

### Supported Providers

- **OpenAI**: `provider="openai"`, models like `gpt-4-turbo`, `gpt-4o`
- **Anthropic**: `provider="anthropic"`, models like `claude-3-opus-20240229`
- **Google**: `provider="google"`, models like `gemini-2.0-flash`

---

## 💻 Usage Example

```python
import asyncio
from argus.config import AgentConfig, ModelConfig
from argus.orchestrator import Orchestrator

async def main():
    config = AgentConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-2.0-flash",
            api_key="..."
        ),
        db_connection_str="postgresql://user:password@localhost:5432/my_db"
    )

    orchestrator = Orchestrator(config)
    result = await orchestrator.run("What is the total revenue by region?")
    print(result["final_memo"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Synchronous Usage

```python
from argus.config import AgentConfig, ModelConfig
from argus.orchestrator import Orchestrator

config = AgentConfig(
    model=ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo",
        api_key="..."
    ),
    db_connection_str="postgresql://user:password@localhost:5432/my_db"
)

orchestrator = Orchestrator(config)
result = orchestrator.run_sync("What is the total revenue by region?")
print(result["final_memo"])
```

### Streaming Response

```python
import asyncio
from argus.config import AgentConfig, ModelConfig
from argus.orchestrator import Orchestrator

async def main():
    config = AgentConfig(
        model=ModelConfig(
            provider="google",
            model_name="gemini-2.0-flash",
            api_key="..."
        ),
        db_connection_str="postgresql://user:password@localhost:5432/my_db"
    )

    orchestrator = Orchestrator(config)
    
    async for chunk in orchestrator.stream_run("What is the total revenue by region?"):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔄 Migration from LangGraph Version

If migrating from the LangGraph-based implementation:

1. **Dependencies**: Replace `langgraph`, `langchain`, `langchain-openai` with `agno`
2. **Model Config**: The `ModelConfig` structure remains the same
3. **Orchestrator API**: 
   - `orchestrator.run()` works the same way (async)
   - Added `orchestrator.run_sync()` for synchronous execution
   - `stream_run()` replaces `astream_events()` for streaming

---

## 📁 Project Structure

```
datax/
├── argus/
│   ├── __init__.py
│   ├── config.py          # Configuration classes
│   ├── models.py          # Agno model factory
│   ├── orchestrator.py    # Main orchestrator
│   ├── safety.py          # SQL validation
│   ├── state.py           # State/context models
│   ├── analyst/
│   │   ├── __init__.py
│   │   ├── decomposer.py  # Query decomposition
│   │   ├── executor.py    # SQL execution
│   │   ├── relevance.py   # Relevance checking
│   │   ├── schema.py      # Schema inspection
│   │   └── sql_gen.py     # SQL generation
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connector.py   # Base connectors
│   │   └── advanced_connectors.py  # MongoDB, ChromaDB
│   └── synthesizer/
│       ├── __init__.py
│       └── memo.py        # Executive memo synthesis
├── pyproject.toml
└── README.md
```

---

## 🔗 Why Agno?

Compared to LangGraph:

- **Simpler Architecture**: Agno uses straightforward Agent and Workflow patterns
- **Built-in Learning**: Agents can remember and improve over time
- **Cleaner Code**: Less boilerplate for common patterns
- **Multiple Providers**: Native support for OpenAI, Anthropic, Google, and more
- **Structured Output**: First-class support for Pydantic models as output schemas

---

## 📘 Full Documentation

For detailed information on architecture, advanced configurations, and multi-source setups, see the [Full Documentation](./argus/DOCUMENTATION.md).

For Agno framework documentation:
- [Agno Documentation](https://docs.agno.com)
- [Agno Cookbook](https://github.com/agno-agi/agno/tree/main/cookbook)
