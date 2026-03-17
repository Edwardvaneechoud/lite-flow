from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSchema:
    """Schema definition for a single column.

    Stores the column's name, Python type, and nullability.
    Immutable (frozen dataclass) so it can be used as a dict key.

    Attributes:
        name: Column name.
        dtype: Python type of the column values.
        nullable: Whether the column may contain None values.

    Examples:
        >>> cs = ColumnSchema("age", int, nullable=False)
        >>> cs.name
        'age'
        >>> cs.dtype
        <class 'int'>
    """

    name: str
    dtype: type = str
    nullable: bool = True

    def with_name(self, name: str) -> ColumnSchema:
        """Return a copy with a different column name."""
        return ColumnSchema(name, self.dtype, self.nullable)

    def with_dtype(self, dtype: type) -> ColumnSchema:
        """Return a copy with a different data type."""
        return ColumnSchema(self.name, dtype, self.nullable)

    def with_nullable(self, nullable: bool) -> ColumnSchema:
        """Return a copy with a different nullability flag."""
        return ColumnSchema(self.name, self.dtype, nullable)

    def __repr__(self):
        n = '?' if self.nullable else ''
        return f'{self.name}: {self.dtype.__name__}{n}'


class LazySchema:
    """Schema for a Floe or plan node, mapping column names to their types.

    A LazySchema propagates through the query plan without touching data,
    so you can inspect the output schema of any pipeline instantly.

    Examples:
        >>> schema = LazySchema.from_dicts([{"name": "Alice", "age": 30}])
        >>> schema.column_names
        ['name', 'age']
        >>> schema.dtypes
        {'name': <class 'str'>, 'age': <class 'int'>}
    """

    __slots__ = ('_columns',)

    def __init__(self, columns: dict[str, ColumnSchema] | None = None):
        """Initialize a LazySchema.

        Args:
            columns: Mapping of column names to ColumnSchema objects.
                If None, creates an empty schema.
        """
        self._columns: dict[str, ColumnSchema] = columns or {}

    @property
    def column_names(self) -> list[str]:
        """List of column names in order."""
        return list(self._columns.keys())

    @property
    def dtypes(self) -> dict[str, type]:
        """Mapping of column names to their Python types."""
        return {n: c.dtype for n, c in self._columns.items()}

    def __getitem__(self, name: str) -> ColumnSchema:
        return self._columns[name]

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __len__(self) -> int:
        return len(self._columns)

    def __iter__(self):
        return iter(self._columns.values())

    def __eq__(self, other):
        if isinstance(other, LazySchema):
            return self._columns == other._columns
        return NotImplemented

    def select(self, columns: list[str]) -> LazySchema:
        """Return a new schema with only the specified columns.

        Args:
            columns: Column names to keep.

        Examples:
            >>> schema = LazySchema({"a": ColumnSchema("a", int), "b": ColumnSchema("b", str)})
            >>> schema.select(["a"]).column_names
            ['a']
        """
        return LazySchema({n: self._columns[n] for n in columns if n in self._columns})

    def drop(self, columns: list[str]) -> LazySchema:
        """Return a new schema without the specified columns.

        Args:
            columns: Column names to remove.

        Examples:
            >>> schema = LazySchema({"a": ColumnSchema("a", int), "b": ColumnSchema("b", str)})
            >>> schema.drop(["b"]).column_names
            ['a']
        """
        drop_set = set(columns)
        return LazySchema({n: c for n, c in self._columns.items() if n not in drop_set})

    def rename(self, mapping: dict[str, str]) -> LazySchema:
        """Return a new schema with columns renamed according to the mapping.

        Args:
            mapping: Old name to new name mapping.

        Examples:
            >>> schema = LazySchema({"a": ColumnSchema("a", int)})
            >>> schema.rename({"a": "x"}).column_names
            ['x']
        """
        cols = {}
        for name, col in self._columns.items():
            new_name = mapping.get(name, name)
            cols[new_name] = col.with_name(new_name)
        return LazySchema(cols)

    def merge(self, other: LazySchema, suffix: str = 'right_') -> LazySchema:
        """Merge two schemas, prefixing duplicate column names.

        Args:
            other: Schema to merge in.
            suffix: Prefix added to duplicate column names from the other schema.

        Examples:
            >>> s1 = LazySchema({"id": ColumnSchema("id", int), "a": ColumnSchema("a", str)})
            >>> s2 = LazySchema({"id": ColumnSchema("id", int), "b": ColumnSchema("b", str)})
            >>> s1.merge(s2).column_names
            ['id', 'a', 'right_id', 'b']
        """
        cols = dict(self._columns)
        for name, col in other._columns.items():
            if name in cols:
                new_name = suffix + name
                cols[new_name] = col.with_name(new_name)
            else:
                cols[name] = col
        return LazySchema(cols)

    def with_column(self, name: str, dtype: type, nullable: bool = True) -> LazySchema:
        """Return a new schema with an added or replaced column.

        Args:
            name: Column name.
            dtype: Python type for the column.
            nullable: Whether the column may contain None.

        Examples:
            >>> schema = LazySchema({"a": ColumnSchema("a", int)})
            >>> schema.with_column("b", str).column_names
            ['a', 'b']
        """
        cols = dict(self._columns)
        cols[name] = ColumnSchema(name, dtype, nullable)
        return LazySchema(cols)

    def with_dtype(self, column: str, dtype: type) -> LazySchema:
        """Return a new schema with the type of one column changed.

        Args:
            column: Column name to change.
            dtype: New Python type.
        """
        cols = dict(self._columns)
        if column in cols:
            cols[column] = cols[column].with_dtype(dtype)
        return LazySchema(cols)

    @classmethod
    def from_data(cls, columns: list[str], rows: list[tuple]) -> LazySchema:
        """Infer a schema from column names and a sample of tuple rows.

        Args:
            columns: Column names.
            rows: Sample rows as tuples (up to 1000 are inspected).

        Examples:
            >>> schema = LazySchema.from_data(["x", "y"], [(1, "hello"), (2, None)])
            >>> schema.dtypes
            {'x': <class 'int'>, 'y': <class 'str'>}
            >>> schema["y"].nullable
            True
        """
        if not rows:
            return cls({n: ColumnSchema(n) for n in columns})
        sample = rows[:min(1000, len(rows))]
        cols = {}
        for i, name in enumerate(columns):
            dtype = str
            nullable = False
            for row in sample:
                val = row[i]
                if val is None:
                    nullable = True
                elif dtype is str:
                    dtype = type(val)
            cols[name] = ColumnSchema(name, dtype, nullable)
        return cls(cols)

    @classmethod
    def from_dicts(cls, data: list[dict]) -> LazySchema:
        """Infer a schema from a sample of dict rows.

        Args:
            data: Sample rows as dicts (up to 1000 are inspected).

        Examples:
            >>> schema = LazySchema.from_dicts([{"name": "Alice", "age": 30}])
            >>> schema.column_names
            ['name', 'age']
            >>> schema.dtypes
            {'name': <class 'str'>, 'age': <class 'int'>}
        """
        if not data:
            return cls()
        cols = {}
        sample = data[:min(1000, len(data))]
        for key in data[0].keys():
            dtype = str
            nullable = False
            for row in sample:
                val = row.get(key)
                if val is None:
                    nullable = True
                elif dtype is str:
                    dtype = type(val)
            cols[key] = ColumnSchema(key, dtype, nullable)
        return cls(cols)

    def __repr__(self):
        if not self._columns:
            return 'Schema(empty)'
        lines = [f'  {col}' for col in self._columns.values()]
        return 'Schema(\n' + '\n'.join(lines) + '\n)'

    def _repr_short(self) -> str:
        parts = [f'{c.name}:{c.dtype.__name__}' for c in self._columns.values()]
        return '{' + ', '.join(parts) + '}'
