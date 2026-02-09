from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import sqlglot
from sqlglot import exp


class SQLValidator:
    def __init__(self, max_tables: int = 8, max_joins: int = 8) -> None:
        """
        Basic SQL safety and schema validator.

        max_tables/max_joins are soft safety limits to guard against runaway queries.
        """
        self.max_tables = max_tables
        self.max_joins = max_joins

    def validate(
        self,
        sql: str,
        dialect: str = "duckdb",
        schema_info: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[bool, str]:
        """
        Validates SQL syntax, strict read-only safety, and optionally schema existence.

        schema_info: { table_name: [col1, col2, ...] }
        Returns: (is_valid, error_message)
        """
        try:
            # Parse
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            return False, f"Syntax Error: {str(e)}"

        # 1. Safety Check: Block mutations and DDL
        forbidden_types = []
        for type_name in [
            "Drop",
            "Delete",
            "Update",
            "Insert",
            "Create",
            "Merge",
            "Commit",
            "Rollback",
            "AlterTable",
            "AlterColumn",
            "TruncateTable",
        ]:
            if hasattr(exp, type_name):
                forbidden_types.append(getattr(exp, type_name))

        if parsed.find(tuple(forbidden_types)):
            return (
                False,
                "Security Violation: Query contains forbidden mutation or DDL statements.",
            )

        # Fallback string check
        sql_upper = sql.upper()
        for cmd in [
            "GRANT ",
            "REVOKE ",
            "DROP ",
            "DELETE ",
            "UPDATE ",
            "INSERT ",
            "CREATE ",
            "ALTER ",
            "TRUNCATE ",
            "MERGE ",
        ]:
            if (
                sql_upper.startswith(cmd)
                or f"\n{cmd}" in sql_upper
                or f";{cmd}" in sql_upper
            ):
                return False, f"Security Violation: Query attempts {cmd.strip()} operation."

        # 2. Complexity heuristics
        tables = list(parsed.find_all(exp.Table))
        joins = list(parsed.find_all(exp.Join))

        if len(tables) > self.max_tables:
            return (
                False,
                f"Safety Violation: Query references too many tables ({len(tables)} > {self.max_tables}).",
            )

        if len(joins) > self.max_joins:
            return (
                False,
                f"Safety Violation: Query uses too many joins ({len(joins)} > {self.max_joins}).",
            )

        # 3. Schema Verification (if provided)
        if schema_info:
            # Normalize schema keys to lowercase for robust matching
            schema_lower = {
                k.lower(): [c.lower() for c in v] for k, v in schema_info.items()
            }

            # Collect derived column aliases (e.g. COUNT(*) AS flight_count)
            alias_names = {
                alias.alias.lower()
                for alias in parsed.find_all(exp.Alias)
                if getattr(alias, "alias", None)
            }

            # Find all CTE names to ignore them as "tables"
            ctes = []
            for cte in parsed.find_all(exp.CTE):
                ctes.append(cte.alias.lower())

            # Check Tables
            query_tables = []
            for table in parsed.find_all(exp.Table):
                t_name = table.name.lower()
                if t_name not in ctes:
                    if t_name not in schema_lower:
                        return False, f"Schema Error: Table '{t_name}' does not exist."
                    query_tables.append(t_name)

            # Check Columns (best effort)
            for column in parsed.find_all(exp.Column):
                col_name = column.name.lower()
                col_table = column.table.lower()

                # Skip validation for derived column aliases (e.g. used in ORDER BY / GROUP BY)
                if col_name in alias_names:
                    continue

                if col_table:
                    # If aliased table, we'd need to resolve alias.
                    # For now, if col_table is in schema or is a query table, check it.
                    if col_table in schema_lower:
                        if col_name not in schema_lower[col_table]:
                            return (
                                False,
                                f"Schema Error: Column '{col_name}' does not exist in table '{col_table}'.",
                            )
                else:
                    # Unqualified column. Check if it exists in ANY table involved in the query.
                    exists = False
                    for t in query_tables:
                        if col_name in schema_lower.get(t, []):
                            exists = True
                            break

                    # If not in query tables, maybe it's a CTE column or something else?
                    # We only error if we are sure it's not in any known physical table.
                    if query_tables and not exists:
                        return (
                            False,
                            f"Schema Error: Column '{col_name}' not found in tables {query_tables}.",
                        )

        return True, ""
