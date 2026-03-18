# Query Plan Nodes

Internal plan node classes that form the query execution tree.  Each node implements the volcano / iterator model — data flows upward through the tree as parent nodes pull rows from their children.

These classes are not part of the public API but are documented here for users who want to understand or extend the query engine internals.

---

## PlanNode

::: pyfloe.plan.PlanNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 3
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
      members_order: source

---

## Data Source Nodes

### ScanNode

::: pyfloe.plan.ScanNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### IteratorSourceNode

::: pyfloe.plan.IteratorSourceNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Transform Nodes

### ProjectNode

::: pyfloe.plan.ProjectNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### FilterNode

::: pyfloe.plan.FilterNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### WithColumnNode

::: pyfloe.plan.WithColumnNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### ApplyNode

::: pyfloe.plan.ApplyNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### RenameNode

::: pyfloe.plan.RenameNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### SortNode

::: pyfloe.plan.SortNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Join Nodes

### JoinNode

::: pyfloe.plan.JoinNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### SortedMergeJoinNode

::: pyfloe.plan.SortedMergeJoinNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Aggregation Nodes

### AggNode

::: pyfloe.plan.AggNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### SortedAggNode

::: pyfloe.plan.SortedAggNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Reshape Nodes

### ExplodeNode

::: pyfloe.plan.ExplodeNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### UnpivotNode

::: pyfloe.plan.UnpivotNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

### PivotNode

::: pyfloe.plan.PivotNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Combine Nodes

### UnionNode

::: pyfloe.plan.UnionNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Window Nodes

### WindowNode

::: pyfloe.plan.WindowNode
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Optimizer

::: pyfloe.Optimizer
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 3
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
