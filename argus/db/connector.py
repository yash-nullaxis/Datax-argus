from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import sqlalchemy
from sqlalchemy import create_engine, inspect, text

from ..safety import SQLValidator


class DBConnector(ABC):
    @abstractmethod
    def get_schema_info(self) -> str:
        """Return schema information as a string."""
        raise NotImplementedError

    @abstractmethod
    def get_schema_dict(self) -> Dict[str, List[str]]:
        """Return schema as a mapping of table names to column lists."""
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return list-of-dict results."""
        raise NotImplementedError


class SqlAlchemyConnector(DBConnector):
    def __init__(self, connection_str: str):
        self.engine = create_engine(connection_str)

    def get_schema_info(self) -> str:
        inspector = inspect(self.engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            col_names = [c['name'] for c in columns]
            
            # Profiling logic
            row_count = 0
            null_counts = {}
            top_values = {}
            
            try:
                with self.engine.connect() as conn:
                    # 1. Total Row Count and Null Counts in one pass
                    # Construct: SELECT COUNT(*), COUNT(col1), COUNT(col2) ...
                    agg_cols = [text("COUNT(*) as total_rows")]
                    for col in col_names:
                        # Use quoted name for safety
                        agg_cols.append(text(f'COUNT("{col}") as "count_{col}"'))
                    
                    agg_query = text(f'SELECT {", ".join([str(c) for c in agg_cols])} FROM "{table_name}"')
                    agg_res = conn.execute(agg_query).mappings().first()
                    
                    row_count = agg_res["total_rows"]
                    for col in col_names:
                        non_null = agg_res[f"count_{col}"]
                        null_counts[col] = row_count - non_null

                    # 2. Top Values for each column (limited to a few samples to stay efficient)
                    for col in col_names:
                        try:
                            top_query = text(f'SELECT "{col}" as val, COUNT(*) as cnt FROM "{table_name}" WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY cnt DESC LIMIT 3')
                            top_res = conn.execute(top_query).mappings().all()
                            top_values[col] = [f"{r['val']} ({r['cnt']})" for r in top_res]
                        except Exception:
                            top_values[col] = ["Error profile"]

            except Exception as e:
                return f"Error profiling table {table_name}: {e}"

            # Format Output
            col_strs = []
            for c in columns:
                name = c['name']
                null_pct = (null_counts.get(name, 0) / row_count * 100) if row_count > 0 else 0
                top_vals_str = ", ".join(top_values.get(name, []))
                col_strs.append(f"  - {name} ({c['type']}) | Nulls: {null_pct:.1f}% | Top: [{top_vals_str}]")
            
            # Sample Rows
            samples_str = "[]"
            try:
                with self.engine.connect() as conn:
                    res = conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT 3'))
                    samples = [dict(row) for row in res.mappings()]
                    samples_str = str(samples)
            except Exception: pass

            schema_info.append(f"Table: {table_name}\nRows: {row_count}\nColumns:\n" + "\n".join(col_strs) + f"\nSummary Samples: {samples_str}")
        return "\n\n".join(schema_info)

    def get_schema_dict(self) -> Dict[str, List[str]]:
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            schema[table_name] = [c['name'] for c in inspector.get_columns(table_name)]
        return schema

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query with safety validation.
        """
        schema = self.get_schema_dict()
        is_valid, error = SQLValidator().validate(query, schema_info=schema)
        if not is_valid:
            raise ValueError(f"SafetyError: {error}")

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return [dict(row) for row in result.mappings()]


class DuckDBConnector(DBConnector):
    """
    Connector for local files (CSV, Parquet, JSON, Excel) using DuckDB.
    """
    def __init__(self, db_path: str = ":memory:", files: List[str] = None):
        import duckdb
        import pandas as pd
        self.conn = duckdb.connect(db_path)
        if files:
            for file_path in files:
                table_name = file_path.split("/")[-1].split(".")[0]
                if file_path.endswith('.xlsx'):
                    # Load Excel via pandas and register
                    df = pd.read_excel(file_path)
                    self.conn.register(table_name, df)
                else:
                    # CSV, Parquet, JSON
                    self.conn.execute(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM '{file_path}'")

    def get_schema_info(self) -> str:
        # Get all tables/views
        tables = self.conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        schema_info = []
        for table_name in table_names:
            columns = self.conn.execute(f'DESCRIBE "{table_name}"').fetchall()
            col_names = [c[0] for c in columns]
            
            # Profiling Logic
            row_count = 0
            null_counts = {}
            top_values = {}
            
            try:
                # 1. Row Count and Nulls
                agg_list = ["COUNT(*)"]
                for col in col_names:
                    agg_list.append(f'COUNT("{col}")')
                
                agg_res = self.conn.execute(f'SELECT {", ".join(agg_list)} FROM "{table_name}"').fetchone()
                row_count = agg_res[0]
                for i, col in enumerate(col_names):
                    non_null = agg_res[i+1]
                    null_counts[col] = row_count - non_null
                
                # 2. Top Values
                for col in col_names:
                    try:
                        top_res = self.conn.execute(f'SELECT "{col}" as val, COUNT(*) as cnt FROM "{table_name}" WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY cnt DESC LIMIT 3').fetchall()
                        top_values[col] = [f"{r[0]} ({r[1]})" for r in top_res]
                    except Exception:
                        top_values[col] = ["Error"]
            except Exception as e:
                 return f"Error profiling DuckDB table {table_name}: {e}"

            # Format Column Strings
            col_strs = []
            for i, col_data in enumerate(columns):
                name = col_data[0]
                dtype = col_data[1]
                null_pct = (null_counts.get(name, 0) / row_count * 100) if row_count > 0 else 0
                top_vals_str = ", ".join(top_values.get(name, []))
                col_strs.append(f"  - {name} ({dtype}) | Nulls: {null_pct:.1f}% | Top: [{top_vals_str}]")

            # Samples
            samples_str = "[]"
            try:
                sample_rows = self.conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3').fetchall()
                sample_cols = [desc[0] for desc in self.conn.description]
                samples = [dict(zip(sample_cols, row)) for row in sample_rows]
                samples_str = str(samples)
            except Exception: pass

            schema_info.append(f"Table: {table_name}\nRows: {row_count}\nColumns:\n" + "\n".join(col_strs) + f"\nSummary Samples: {samples_str}")
        
        return "\n\n".join(schema_info)

    def get_schema_dict(self) -> Dict[str, List[str]]:
        tables = self.conn.execute("SHOW TABLES").fetchall()
        schema = {}
        for t in tables:
            t_name = t[0]
            cols = self.conn.execute(f'DESCRIBE "{t_name}"').fetchall()
            schema[t_name] = [c[0] for c in cols]
        return schema

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a read-only DuckDB query with safety validation.
        """
        schema = self.get_schema_dict()
        is_valid, error = SQLValidator().validate(
            query, dialect="duckdb", schema_info=schema
        )
        if not is_valid:
            raise ValueError(f"SafetyError: {error}")

        # DuckDB returns list of tuples or we can convert to arrow/pandas/dict
        # Using fetchall() and description to make dicts
        cursor = self.conn.execute(query)
        result = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        return [dict(zip(cols, row)) for row in result]
    
    @property
    def engine(self):
        return None
