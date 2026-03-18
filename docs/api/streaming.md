# Streaming

Functions and classes for streaming data from iterators, generators, and paginated sources.

---

## Functions

### from_iter

::: pyfloe.from_iter
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

### from_chunks

::: pyfloe.from_chunks
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

## Stream

The `Stream` class provides a true single-pass streaming pipeline. Unlike LazyFrame, it compiles transforms into a flat loop for maximum throughput with no plan-tree overhead.

::: pyfloe.Stream
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
