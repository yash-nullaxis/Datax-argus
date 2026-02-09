"""
Schema inspector for database schema introspection and RAG indexing.
"""
import logging
from typing import Dict, Optional, Union
import uuid

from ..db.connector import DBConnector


logger = logging.getLogger(__name__)


class SchemaInspector:
    """Inspects database schemas and provides schema context for LLM."""
    
    def __init__(
        self,
        db: Union[DBConnector, Dict[str, DBConnector]],
        vector_store: Optional[DBConnector] = None,
    ):
        self.db = db
        self.vector_store = vector_store

        # Trigger indexing if RAG is enabled
        if self.vector_store:
            self._index_schemas()

    def get_summary(self, query: str = None) -> str:
        """
        Returns a natural language summary of the database schema with profiling stats.
        If query is provided and RAG is enabled, returns relevant tables only to save context.
        """
        full_summary = self._get_global_summary()

        if query and self.vector_store:
            # Perform RAG retrieval
            try:
                # We assume execute_query on Chroma connector returns relevant doc content
                # ChromaDBConnector returns list of dicts with 'content' and 'metadata'
                results = self.vector_store.execute_query(query)
                if results:
                    context_parts = []
                    for r in results:
                        content = r.get("content", "")
                        meta = r.get("metadata", {})
                        context_parts.append(f"{content}\n(Metadata: {meta})")

                    if context_parts:
                        return (
                            "--- RELEVANT TABLES (RAG FILTERED) ---\n"
                            + "\n\n".join(context_parts)
                        )
            except Exception as e:
                # Fallback to full summary on error
                return f"{full_summary}\n(RAG Error: {e})"

        return full_summary

    def _index_schemas(self):
        """
        Indexes table schemas into the vector store for RAG.
        This parses the text summary and chunks it by table (heuristic).
        """
        try:
            full_text = self._get_global_summary()

            # Split on explicit 'Table:' / 'Collection:' boundaries when present.
            chunks = []
            metadatas = []
            ids = []

            raw_sections = []
            current = []
            for line in full_text.splitlines():
                if line.startswith("Table: ") or line.startswith("Collection: "):
                    if current:
                        raw_sections.append("\n".join(current))
                        current = []
                current.append(line)
            if current:
                raw_sections.append("\n".join(current))

            if not raw_sections:
                raw_sections = full_text.split("\n\n")

            for section in raw_sections:
                if not section.strip():
                    continue

                # Metadata extraction
                source = "unknown"
                table = None
                if "--- SOURCE:" in section:
                    try:
                        source = (
                            section.split("SOURCE:")[1].split("---")[0].strip()
                        )
                    except Exception:
                        pass

                for line in section.splitlines():
                    if line.startswith("Table: "):
                        table = line.replace("Table: ", "").strip()
                        break
                    if line.startswith("Collection: "):
                        table = line.replace("Collection: ", "").strip()
                        break

                chunks.append(section)
                metadatas.append(
                    {
                        "source": source,
                        "table": table,
                        "type": "schema_summary",
                    }
                )
                ids.append(str(uuid.uuid4()))

            # Upsert to Vector Store
            if hasattr(self.vector_store, "add_documents"):
                self.vector_store.add_documents(
                    documents=chunks, metadatas=metadatas, ids=ids
                )

        except Exception as e:
            logger.warning(f"Failed to index schemas for RAG: {e}")

    def _get_global_summary(self) -> str:
        """
        Aggregates schema info from all connectors, including a short structured header.
        """
        if isinstance(self.db, dict):
            # Multi-source mode
            global_summary = []
            for name, connector in self.db.items():
                try:
                    schema_dict = connector.get_schema_dict()
                    header_lines = []
                    for table_name, cols in schema_dict.items():
                        preview_cols = ", ".join(cols[:5])
                        header_lines.append(
                            f"- {table_name}: {preview_cols}"
                        )
                    header = "Tables:\n" + "\n".join(header_lines) if header_lines else "Tables: (none detected)"

                    schema = connector.get_schema_info()
                    global_summary.append(
                        f"--- SOURCE: {name} ---\n{header}\n\n{schema}"
                    )
                except Exception as e:
                    global_summary.append(
                        f"--- SOURCE: {name} ---\nError fetching schema: {e}"
                    )
            return "\n\n".join(global_summary)
        else:
            schema_dict = self.db.get_schema_dict()
            header_lines = []
            for table_name, cols in schema_dict.items():
                preview_cols = ", ".join(cols[:5])
                header_lines.append(f"- {table_name}: {preview_cols}")
            header = "Tables:\n" + "\n".join(header_lines) if header_lines else "Tables: (none detected)"
            return f"{header}\n\n{self.db.get_schema_info()}"
