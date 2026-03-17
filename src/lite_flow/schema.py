from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype: type = str
    nullable: bool = True

    def with_name(self, name: str) -> ColumnSchema:
        return ColumnSchema(name, self.dtype, self.nullable)

    def with_dtype(self, dtype: type) -> ColumnSchema:
        return ColumnSchema(self.name, dtype, self.nullable)

    def with_nullable(self, nullable: bool) -> ColumnSchema:
        return ColumnSchema(self.name, self.dtype, nullable)

    def __repr__(self):
        n = '?' if self.nullable else ''
        return f'{self.name}: {self.dtype.__name__}{n}'


class LazySchema:
    __slots__ = ('_columns',)

    def __init__(self, columns: dict[str, ColumnSchema] | None = None):
        self._columns: dict[str, ColumnSchema] = columns or {}

    @property
    def column_names(self) -> list[str]:
        return list(self._columns.keys())

    @property
    def dtypes(self) -> dict[str, type]:
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
        return LazySchema({n: self._columns[n] for n in columns if n in self._columns})

    def drop(self, columns: list[str]) -> LazySchema:
        drop_set = set(columns)
        return LazySchema({n: c for n, c in self._columns.items() if n not in drop_set})

    def rename(self, mapping: dict[str, str]) -> LazySchema:
        cols = {}
        for name, col in self._columns.items():
            new_name = mapping.get(name, name)
            cols[new_name] = col.with_name(new_name)
        return LazySchema(cols)

    def merge(self, other: LazySchema, suffix: str = 'right_') -> LazySchema:
        cols = dict(self._columns)
        for name, col in other._columns.items():
            if name in cols:
                new_name = suffix + name
                cols[new_name] = col.with_name(new_name)
            else:
                cols[name] = col
        return LazySchema(cols)

    def with_column(self, name: str, dtype: type, nullable: bool = True) -> LazySchema:
        cols = dict(self._columns)
        cols[name] = ColumnSchema(name, dtype, nullable)
        return LazySchema(cols)

    def with_dtype(self, column: str, dtype: type) -> LazySchema:
        cols = dict(self._columns)
        if column in cols:
            cols[column] = cols[column].with_dtype(dtype)
        return LazySchema(cols)

    @classmethod
    def from_data(cls, columns: list[str], rows: list[tuple]) -> LazySchema:
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
