#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ast-grep-py>=0.31.0",
#     "networkx>=3.0",
# ]
# ///
"""
Apply the refactoring plan to transform the CXI provider codebase.

This script:
1. Reads the refactor plan (refactor_plan.json)
2. Extracts symbol definitions from cxip.h
3. Creates new header files under prov/cxi/include/cxip/
4. Updates source files to include appropriate headers
5. Removes extracted content from cxip.h

The transformation is done in multiple passes:
- Pass 1: Parse and extract all symbol definitions from cxip.h
- Pass 2: Group symbols by target header
- Pass 3: Generate new header files with proper include guards
- Pass 4: Update source file includes
- Pass 5: Clean up cxip.h to only include the new headers
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from ast_grep_py import SgRoot
import networkx as nx


# License header for new files
LICENSE_HEADER = """\
/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */
"""


@dataclass
class ExtractedSymbol:
    """A symbol extracted from the source."""
    name: str
    kind: str  # "function", "type", "macro"
    text: str  # Full text of the declaration/definition
    start_line: int
    end_line: int
    dependencies: list[str] = field(default_factory=list)


@dataclass
class HeaderContent:
    """Content for a new header file."""
    path: str
    symbols: list[ExtractedSymbol] = field(default_factory=list)
    includes: set[str] = field(default_factory=set)
    forward_decls: set[str] = field(default_factory=set)


def load_refactor_plan(path: str | None = None) -> dict:
    """Load the refactor plan JSON from file or stdin if in a pipeline."""
    if path:
        with open(path) as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def get_include_guard(header_path: str) -> tuple[str, str]:
    """Generate include guard macros for a header."""
    # Convert path like "prov/cxi/include/cxip/ep.h" to "_CXIP_EP_H_"
    name = Path(header_path).stem.upper()
    guard = f"_CXIP_{name}_H_"
    return f"#ifndef {guard}\n#define {guard}\n", f"#endif /* {guard} */\n"


def extract_macro_definitions(content: str, macro_names: set[str]) -> dict[str, ExtractedSymbol]:
    """Extract macro definitions using regex (more reliable for preprocessor)."""
    macros = {}
    lines = content.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Match #define MACRO_NAME
        match = re.match(r'^#define\s+(\w+)(?:\(|[^(]|\s|$)', line)
        if match:
            macro_name = match.group(1)
            if macro_name in macro_names:
                # Find the full extent (handle line continuations)
                start_line = i
                end_line = i
                while end_line < len(lines) and lines[end_line].rstrip().endswith('\\'):
                    end_line += 1

                macro_text = '\n'.join(lines[start_line:end_line + 1])
                macros[macro_name] = ExtractedSymbol(
                    name=macro_name,
                    kind="macro",
                    text=macro_text,
                    start_line=start_line + 1,  # 1-indexed
                    end_line=end_line + 1,
                )
        i += 1

    return macros


def extract_type_definitions(root: SgRoot, content: str, type_names: set[str]) -> tuple[dict[str, ExtractedSymbol], dict[str, ExtractedSymbol]]:
    """Extract type definitions using ast-grep.

    Returns:
        Tuple of (enums, other_types) - enums are separated because they
        need to be defined early (before structs that use them as fields).
    """
    enums = {}
    types = {}
    lines = content.split('\n')

    # Extract struct/union/enum definitions
    for kind, type_kind in [("struct_specifier", "struct"),
                            ("union_specifier", "union"),
                            ("enum_specifier", "enum")]:
        for node in root.root().find_all(kind=kind):
            # Get the name
            name_node = None
            for child in node.children():
                if child.kind() == "type_identifier":
                    name_node = child
                    break

            if name_node is None:
                continue

            type_name = name_node.text()
            if type_name not in type_names:
                continue

            # Check if this is a definition (has body)
            has_body = False
            for child in node.children():
                if child.kind() in ("field_declaration_list", "enumerator_list"):
                    has_body = True
                    break

            if not has_body:
                continue  # Skip forward declarations

            # Get the full text including potential typedef wrapper
            parent = node.parent()
            if parent and parent.kind() == "type_definition":
                node_to_extract = parent
            else:
                # Check if this is part of a declaration
                if parent and parent.kind() == "declaration":
                    node_to_extract = parent
                else:
                    node_to_extract = node

            range_info = node_to_extract.range()
            start = range_info.start.line
            end = range_info.end.line

            # Extract the lines, including the semicolon if needed
            text = '\n'.join(lines[start:end + 1])
            if not text.rstrip().endswith(';'):
                # Look for semicolon on next line
                if end + 1 < len(lines) and lines[end + 1].strip() == ';':
                    text += '\n;'
                    end += 1

            sym = ExtractedSymbol(
                name=type_name,
                kind="type",
                text=text,
                start_line=start + 1,
                end_line=end + 1,
            )

            # Enums go to a separate collection
            if type_kind == "enum":
                enums[type_name] = sym
            else:
                types[type_name] = sym

    # Extract typedefs
    for node in root.root().find_all(kind="type_definition"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        # Get the typedef'd name
        type_name = None
        if declarator.kind() == "type_identifier":
            type_name = declarator.text()
        else:
            for child in declarator.children():
                if child.kind() == "type_identifier":
                    type_name = child.text()
                    break

        if type_name is None or type_name not in type_names:
            continue

        # Don't re-add if we already have this from struct extraction
        if type_name in types:
            continue

        # Skip typedefs of enums - they're already extracted into enums dict
        # Check if this typedef contains an enum_specifier
        is_enum_typedef = False
        for child in node.children():
            if child.kind() == "enum_specifier":
                is_enum_typedef = True
                break
        if is_enum_typedef:
            continue

        range_info = node.range()
        start = range_info.start.line
        end = range_info.end.line

        text = '\n'.join(lines[start:end + 1])

        types[type_name] = ExtractedSymbol(
            name=type_name,
            kind="type",
            text=text,
            start_line=start + 1,
            end_line=end + 1,
        )

    return enums, types


def extract_function_declarations(root: SgRoot, content: str, func_names: set[str]) -> tuple[dict[str, ExtractedSymbol], dict[str, ExtractedSymbol]]:
    """Extract function declarations from header.

    Returns:
        Tuple of (non_inline_functions, inline_functions)
        Inline functions are kept separate to be placed after all types are defined.
    """
    non_inline_functions = {}
    inline_functions = {}
    lines = content.split('\n')

    # Find function declarations (not definitions - those have bodies)
    for node in root.root().find_all(kind="declaration"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        # Check if this has a function_declarator
        has_func_decl = False
        check_node = declarator
        while check_node:
            if check_node.kind() == "function_declarator":
                has_func_decl = True
                break
            # Look in children
            found = None
            for child in check_node.children():
                if child.kind() in ("function_declarator", "pointer_declarator"):
                    found = child
                    break
            check_node = found

        if not has_func_decl:
            continue

        # Get the function name
        func_name = find_identifier_in_declarator(declarator)
        if func_name is None or func_name not in func_names:
            continue

        range_info = node.range()
        start = range_info.start.line
        end = range_info.end.line

        text = '\n'.join(lines[start:end + 1])

        non_inline_functions[func_name] = ExtractedSymbol(
            name=func_name,
            kind="function",
            text=text,
            start_line=start + 1,
            end_line=end + 1,
        )

    # Find static inline function definitions - these go to a separate collection
    for node in root.root().find_all(kind="function_definition"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        func_name = find_identifier_in_declarator(declarator)
        if func_name is None or func_name not in func_names:
            continue

        full_text = node.text()
        prefix = full_text.split(func_name)[0] if func_name in full_text else ""

        # Only include if static inline
        if "static" in prefix and "inline" in prefix:
            range_info = node.range()
            start = range_info.start.line
            end = range_info.end.line

            text = '\n'.join(lines[start:end + 1])

            inline_functions[func_name] = ExtractedSymbol(
                name=func_name,
                kind="inline_function",
                text=text,
                start_line=start + 1,
                end_line=end + 1,
            )

    return non_inline_functions, inline_functions


def find_type_references(text: str, all_type_names: set[str]) -> dict[str, str]:
    """Find all type references in a piece of code.

    Returns a dict mapping type_name -> kind ('struct', 'union', 'enum', or 'typedef')
    for types that are referenced but might need forward declarations.

    We detect:
    - struct foo *  -> needs forward decl "struct foo;"
    - union foo *   -> needs forward decl "union foo;"
    - struct foo field; -> needs full definition (embedded field - can't forward declare)
    - enum foo field; -> needs full definition (embedded)
    """
    references = {}

    # Pattern for struct/union/enum references
    # Match: struct/union/enum type_name followed by * (pointer) or identifier (field)
    for kind in ['struct', 'union', 'enum']:
        # Find all occurrences of "struct typename" or "union typename" etc
        pattern = rf'\b{kind}\s+(\w+)\s*(\*?)'
        for match in re.finditer(pattern, text):
            type_name = match.group(1)
            is_pointer = bool(match.group(2))
            if type_name in all_type_names:
                # For pointers, we can use forward declarations
                # For embedded fields, we need the full type
                if is_pointer:
                    if type_name not in references:
                        references[type_name] = kind
                # If it's not a pointer, it's an embedded field - mark as needing full type
                # We'll handle this differently (can't forward declare)

    return references


def find_embedded_type_references(text: str, all_type_names: set[str]) -> set[str]:
    """Find types that are embedded (not pointers) and need full definitions.

    These CANNOT be forward declared - the including header must come first.
    """
    embedded = set()

    # Pattern for embedded struct/union/enum fields (not pointers)
    # Match: struct/union/enum type_name identifier; (without *)
    for kind in ['struct', 'union', 'enum']:
        # Look for embedded fields: "struct foo bar;" or "struct foo bar[N];"
        # NOT "struct foo *bar;" (pointer)
        pattern = rf'\b{kind}\s+(\w+)\s+(?!\*)\w+[\s\[\];]'
        for match in re.finditer(pattern, text):
            type_name = match.group(1)
            if type_name in all_type_names:
                embedded.add(type_name)

    return embedded


def extract_function_pointer_typedefs(content: str) -> dict[str, ExtractedSymbol]:
    """Extract function pointer typedefs that may not be caught by ast-grep.

    These have the form: typedef returntype (*name)(params);
    """
    typedefs = {}
    lines = content.split('\n')

    # Pattern: typedef <type> (*<name>)(<params>);
    pattern = r'typedef\s+\w+\s+\(\*(\w+)\)\s*\([^)]*\)\s*;'

    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            typedefs[name] = ExtractedSymbol(
                name=name,
                kind="type",
                text=line,
                start_line=i + 1,
                end_line=i + 1,
            )

    return typedefs


def generate_forward_declarations(type_refs: dict[str, str]) -> list[str]:
    """Generate forward declaration statements for the given type references.

    Args:
        type_refs: dict mapping type_name -> kind ('struct', 'union', 'enum')

    Returns:
        List of forward declaration strings
    """
    decls = []
    for type_name, kind in sorted(type_refs.items()):
        if kind in ('struct', 'union'):
            decls.append(f"{kind} {type_name};")
        # Note: enums can't be forward declared in C
    return decls


def find_identifier_in_declarator(node) -> str | None:
    """Recursively find the identifier in a declarator."""
    if node.kind() == "identifier":
        return node.text()

    # Check field access first
    field_result = node.field("declarator")
    if field_result:
        result = find_identifier_in_declarator(field_result)
        if result:
            return result

    # Then check children
    for child in node.children():
        if child.kind() == "identifier":
            return child.text()
        elif child.kind() in ("function_declarator", "pointer_declarator",
                              "array_declarator", "parenthesized_declarator"):
            result = find_identifier_in_declarator(child)
            if result:
                return result

    return None


def detect_required_includes(text: str) -> list[str]:
    """Detect which standard/ofi includes are needed based on types used in the code.

    Returns a list of include directives in the correct order.
    """
    includes = []

    # Map of type patterns to their required includes
    # Order matters - more fundamental includes should come first
    # NOTE: These includes are for documentation purposes since the wrapper
    # cxip.h already includes all external dependencies. But they help
    # make each split header more self-documenting.
    include_checks = [
        # Standard C headers
        (r'\b(uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t|uintptr_t)\b',
         '<stdint.h>'),
        (r'\bsize_t\b', '<stddef.h>'),
        (r'\bbool\b', '<stdbool.h>'),

        # POSIX headers
        (r'\bpthread_(rwlock_t|mutex_t|cond_t|t)\b', '<pthread.h>'),
        (r'\bsem_t\b', '<semaphore.h>'),

        # OFI headers - order matters for dependencies
        # Note: ofi_spin_t and ofi_mutex_t are both defined in ofi_lock.h
        (r'\b(dlist_entry|slist_entry|slist|dlist_ts)\b', '<ofi_list.h>'),
        (r'\bofi_atomic32_t\b', '<ofi_atom.h>'),
        (r'\b(ofi_spin_t|ofi_mutex_t)\b', '<ofi_lock.h>'),
    ]

    seen = set()
    for pattern, include in include_checks:
        if include not in seen and re.search(pattern, text):
            includes.append(f'#include {include}')
            seen.add(include)

    return includes


def generate_header_file(header: HeaderContent, all_type_names: set[str],
                         types_defined_in_header: set[str],
                         enum_names_in_enums_h: set[str]) -> str:
    """Generate the content of a new header file.

    Note: This generates headers WITHOUT includes. The main cxip.h will
    include everything in the correct order to handle dependencies.
    Individual headers are not meant to be standalone.

    IMPORTANT: Inline functions are NOT included in split headers - they
    remain in cxip.h after all type definitions, because they often
    access struct members from multiple modules.

    Args:
        header: The header content to generate
        all_type_names: Set of all known type names across all headers
        types_defined_in_header: Set of type names defined in THIS header
        enum_names_in_enums_h: Set of enum names that are in enums.h (skip these)
    """
    guard_start, guard_end = get_include_guard(header.path)

    lines = []
    lines.append(LICENSE_HEADER)
    lines.append(guard_start)
    lines.append("")

    # Group symbols by kind, preserving original source order (by start_line)
    # NOTE: inline_function kind is excluded - those stay in cxip.h
    # NOTE: Skip enums that are already in enums.h
    macros = [s for s in header.symbols if s.kind == "macro"]
    types = [s for s in header.symbols if s.kind == "type" and s.name not in enum_names_in_enums_h]
    functions = [s for s in header.symbols if s.kind == "function"]

    # Sort by original source line to preserve dependency order
    macros.sort(key=lambda s: s.start_line)
    types.sort(key=lambda s: s.start_line)
    functions.sort(key=lambda s: s.start_line)

    # Collect all text to detect required includes
    all_symbol_text = '\n'.join(sym.text for sym in header.symbols)
    required_includes = detect_required_includes(all_symbol_text)
    if required_includes:
        for inc in required_includes:
            lines.append(inc)
        lines.append("")

    # Compute forward declarations needed for function declarations
    # (types used as pointers in function signatures)
    all_text = '\n'.join(sym.text for sym in functions)
    type_refs = find_type_references(all_text, all_type_names)

    # Also scan type definitions for function pointer members
    # These can have struct/union pointers in their parameter lists
    # e.g., int (*callback)(struct cxip_req *req, const union c_event *event);
    type_text = '\n'.join(sym.text for sym in types)
    type_refs_from_types = find_type_references(type_text, all_type_names)
    type_refs.update(type_refs_from_types)

    # Remove types that are defined in this header (no forward decl needed)
    for defined_type in types_defined_in_header:
        type_refs.pop(defined_type, None)

    # Generate forward declarations
    forward_decls = generate_forward_declarations(type_refs)
    if forward_decls:
        lines.append("/* Forward declarations */")
        for decl in forward_decls:
            lines.append(decl)
        lines.append("")

    # Add macros first
    if macros:
        lines.append("/* Macros */")
        for sym in macros:
            lines.append(sym.text)
            lines.append("")

    # Add types - preserve original order for dependencies
    if types:
        lines.append("/* Type definitions */")
        for sym in types:
            lines.append(sym.text)
            lines.append("")

    # Add function declarations (non-inline only)
    if functions:
        lines.append("/* Function declarations */")
        for sym in functions:
            lines.append(sym.text)
            lines.append("")

    lines.append(guard_end)

    return '\n'.join(lines)


def generate_mr_lac_cache_header(mr_lac_cache_sym: ExtractedSymbol) -> str:
    """Generate a dedicated header for cxip_mr_lac_cache to break the mr.h/ctrl.h cycle.

    This struct is used by ctrl.h but defined in mr.h, creating a circular dependency.
    By moving it to its own header that comes before both, we break the cycle.
    """
    guard_start, guard_end = get_include_guard("prov/cxi/include/cxip/mr_lac_cache.h")

    lines = []
    lines.append(LICENSE_HEADER)
    lines.append(guard_start)
    lines.append("")
    lines.append("/* cxip_mr_lac_cache type definition */")
    lines.append("/* This is in a separate header to break the circular dependency between mr.h and ctrl.h */")
    lines.append("")
    lines.append("/* Forward declarations */")
    lines.append("struct cxip_ctrl_req;")
    lines.append("")
    lines.append(mr_lac_cache_sym.text)
    lines.append("")
    lines.append(guard_end)

    return '\n'.join(lines)


def find_macro_references(text: str, all_macro_names: set[str]) -> set[str]:
    """Find all macros referenced in the code.

    Macros can be used as:
    - Bit-field widths: uint32_t field:MACRO_NAME;
    - Array sizes: type arr[MACRO_NAME];
    - Initializers: .field = MACRO_NAME
    - etc.
    """
    referenced = set()
    for macro_name in all_macro_names:
        # Look for the macro name as a standalone token
        pattern = rf'\b{re.escape(macro_name)}\b'
        if re.search(pattern, text):
            referenced.add(macro_name)
    return referenced


def generate_enums_header(enums: list[ExtractedSymbol]) -> str:
    """Generate a dedicated enums.h header with all enum definitions.

    This header is included first because enums are needed by many structs
    (like cxip_environment) that embed enum fields.
    """
    guard_start, guard_end = get_include_guard("prov/cxi/include/cxip/enums.h")

    lines = []
    lines.append(LICENSE_HEADER)
    lines.append(guard_start)
    lines.append("")
    lines.append("/* All enum type definitions */")
    lines.append("/* Included first because many structs embed enum fields */")
    lines.append("")

    # Sort by original source line to preserve order
    sorted_enums = sorted(enums, key=lambda s: s.start_line)
    for sym in sorted_enums:
        lines.append(sym.text)
        lines.append("")

    lines.append(guard_end)

    return '\n'.join(lines)


def build_header_dependency_graph(
    headers: dict[str, 'HeaderContent'],
    type_to_header: dict[str, str],
    macro_to_header: dict[str, str],
    all_type_names: set[str],
    all_macro_names: set[str]
) -> nx.DiGraph:
    """Build a directed graph of header dependencies based on embedded type and macro usage.

    For each header, we analyze which types it uses as embedded fields (not pointers)
    and which macros it references. If a type is embedded or a macro is used,
    the header defining that symbol must be included first.

    Args:
        headers: Dict mapping header path -> HeaderContent
        type_to_header: Dict mapping type name -> header path where it's defined
        macro_to_header: Dict mapping macro name -> header path where it's defined
        all_type_names: Set of all known type names
        all_macro_names: Set of all known macro names

    Returns:
        A directed graph where edge A->B means A must be included before B
    """
    G = nx.DiGraph()

    # Add all headers as nodes
    for header_path in headers:
        header_name = Path(header_path).name
        G.add_node(header_name)

    # Always include enums.h first
    G.add_node("enums.h")

    # For each header, find embedded type and macro dependencies
    for header_path, header_content in headers.items():
        header_name = Path(header_path).name

        # Collect all text from types defined in this header
        type_texts = [s.text for s in header_content.symbols if s.kind == "type"]
        all_text = '\n'.join(type_texts)

        # Find embedded type references (types used as fields, not pointers)
        embedded_refs = find_embedded_type_references(all_text, all_type_names)

        for embedded_type in embedded_refs:
            # Find which header defines this type
            if embedded_type in type_to_header:
                dep_header_path = type_to_header[embedded_type]
                dep_header_name = Path(dep_header_path).name

                # Don't add self-edges
                if dep_header_name != header_name:
                    # Add edge: dependency must come before this header
                    G.add_edge(dep_header_name, header_name)
                    print(f"  Dependency: {header_name} embeds type from {dep_header_name} ({embedded_type})",
                          file=sys.stderr)

        # Find macro references (used in bit-fields, array sizes, etc.)
        macro_refs = find_macro_references(all_text, all_macro_names)
        # Also include macros defined in this header (to exclude self-refs)
        macros_in_this_header = {s.name for s in header_content.symbols if s.kind == "macro"}

        for macro_name in macro_refs:
            if macro_name in macros_in_this_header:
                continue  # Skip self-references
            if macro_name in macro_to_header:
                dep_header_path = macro_to_header[macro_name]
                dep_header_name = Path(dep_header_path).name

                if dep_header_name != header_name:
                    G.add_edge(dep_header_name, header_name)
                    print(f"  Dependency: {header_name} uses macro from {dep_header_name} ({macro_name})",
                          file=sys.stderr)

    # enums.h should come before everything else
    for node in G.nodes():
        if node != "enums.h":
            G.add_edge("enums.h", node)

    # mr_lac_cache.h must come before ctrl.h (to break the circular dependency)
    # ctrl.h embeds cxip_mr_lac_cache which is defined in mr_lac_cache.h
    G.add_node("mr_lac_cache.h")
    G.add_edge("enums.h", "mr_lac_cache.h")  # enums must come first
    # mr_lac_cache.h embeds union cxip_match_bits from msg.h
    if "msg.h" in G.nodes():
        G.add_edge("msg.h", "mr_lac_cache.h")
        print(f"  Dependency: mr_lac_cache.h embeds union from msg.h (cxip_match_bits)", file=sys.stderr)
    if "ctrl.h" in G.nodes():
        G.add_edge("mr_lac_cache.h", "ctrl.h")
        print(f"  Dependency: ctrl.h needs mr_lac_cache.h (cxip_mr_lac_cache)", file=sys.stderr)
    # Also mr.h might reference it
    if "mr.h" in G.nodes():
        G.add_edge("mr_lac_cache.h", "mr.h")
        print(f"  Dependency: mr.h needs mr_lac_cache.h (cxip_mr_lac_cache)", file=sys.stderr)

    return G


def compute_header_order(G: nx.DiGraph, fallback_order: list[str]) -> list[str]:
    """Compute the topological order of headers.

    Args:
        G: Dependency graph where edge A->B means A must come before B
        fallback_order: Default order to use for headers not in the graph

    Returns:
        List of header names in correct dependency order
    """
    try:
        # Use topological sort to get correct order
        topo_order = list(nx.topological_sort(G))
        print(f"Topological order computed: {len(topo_order)} headers", file=sys.stderr)

        # Add any headers from fallback_order that aren't in the graph
        result = []
        seen = set()
        for h in topo_order:
            if h not in seen:
                result.append(h)
                seen.add(h)

        for h in fallback_order:
            if h not in seen:
                result.append(h)
                seen.add(h)

        return result

    except nx.NetworkXUnfeasible as e:
        # Cycle detected - report it and fall back to manual order
        print(f"WARNING: Cycle detected in dependency graph: {e}", file=sys.stderr)
        try:
            cycle = nx.find_cycle(G)
            print(f"  Cycle: {cycle}", file=sys.stderr)
        except nx.NetworkXNoCycle:
            pass
        return fallback_order


def generate_wrapper_cxip_h(headers: dict[str, 'HeaderContent'],
                           type_to_header: dict[str, str],
                           macro_to_header: dict[str, str],
                           all_type_names: set[str],
                           all_macro_names: set[str],
                           original_content: str,
                           inline_functions: list[ExtractedSymbol],
                           func_ptr_typedefs: list[ExtractedSymbol] = None) -> str:
    """Generate a new cxip.h that includes all the split headers.

    This preserves the original includes and external dependencies,
    then includes all the new split headers, followed by inline functions.

    The structure is:
    1. License header
    2. Include guard
    3. External includes (ofi, libcxi, etc.)
    4. Split headers (types, macros, non-inline function declarations)
    5. Inline function definitions (need all types to be defined first)
    6. End guard
    """
    lines = []

    # Extract the original license and includes section
    original_lines = original_content.split('\n')

    # Copy license header
    lines.append("/*")
    for line in original_lines[1:10]:  # Get the license block
        if line.startswith(" */"):
            lines.append(line)
            break
        lines.append(line)

    lines.append("")
    lines.append("#ifndef _CXIP_PROV_H_")
    lines.append("#define _CXIP_PROV_H_")
    lines.append("")

    # Copy all the original system/library includes
    in_includes = False
    for line in original_lines:
        if line.startswith("#include"):
            in_includes = True
            # Skip only the new split headers (cxip/), keep other cxip includes
            if "cxip/" not in line:
                lines.append(line)
        elif in_includes and line.strip() == "":
            lines.append("")
        elif in_includes and not line.startswith("#"):
            break

    # Add function pointer typedefs that aren't in the plan
    # These need to come before the split headers that use them
    if func_ptr_typedefs:
        lines.append("")
        lines.append("/* Forward declarations for function pointer typedef parameters */")
        # Extract struct names referenced in the typedefs
        for typedef in func_ptr_typedefs:
            for match in re.finditer(r'struct\s+(\w+)', typedef.text):
                struct_name = match.group(1)
                lines.append(f"struct {struct_name};")
        lines.append("")
        lines.append("/* Function pointer typedefs (needed by split headers) */")
        sorted_typedefs = sorted(func_ptr_typedefs, key=lambda s: s.start_line)
        for typedef in sorted_typedefs:
            lines.append(typedef.text)
        lines.append("")

    # Add extern declarations for global variables used in source files
    lines.append("/* Extern declarations for global variables */")
    lines.append("extern struct cxip_environment cxip_env;")
    lines.append("extern struct fi_provider cxip_prov;")
    lines.append("extern struct util_prov cxip_util_prov;")
    lines.append("extern char cxip_prov_name[];")
    lines.append("extern struct fi_fabric_attr cxip_fabric_attr;")
    lines.append("extern struct fi_domain_attr cxip_domain_attr;")
    lines.append("extern bool cxip_collectives_supported;")
    lines.append("extern int sc_page_size;")
    lines.append("extern struct slist cxip_if_list;")
    lines.append("")
    lines.append("/* Coll trace globals used by inline trace functions */")
    lines.append("extern bool cxip_coll_trace_muted;")
    lines.append("extern bool cxip_coll_trace_append;")
    lines.append("extern bool cxip_coll_trace_linebuf;")
    lines.append("extern int cxip_coll_trace_rank;")
    lines.append("extern int cxip_coll_trace_numranks;")
    lines.append("extern FILE *cxip_coll_trace_fid;")
    lines.append("extern bool cxip_coll_prod_trace_initialized;")
    lines.append("extern uint64_t cxip_coll_trace_mask;")
    lines.append("")

    lines.append("/* Split headers - types, macros, and function declarations */")

    # Build dependency graph and compute topological order
    print("\nBuilding header dependency graph...", file=sys.stderr)
    dep_graph = build_header_dependency_graph(headers, type_to_header, macro_to_header,
                                              all_type_names, all_macro_names)

    # Fallback order in case of cycles or issues
    # NOTE: mr_lac_cache.h breaks the circular dependency between mr.h and ctrl.h
    # by defining cxip_mr_lac_cache in a separate header that comes before both.
    fallback_order = [
        "enums.h", "addr.h", "common.h", "log.h", "env.h", "if.h",
        "iomm.h", "evtq.h", "cmdq.h", "pte.h", "eq.h", "cq.h", "cntr.h",
        "msg.h",           # Must be before mr_lac_cache.h (defines cxip_match_bits)
        "mr_lac_cache.h",  # Contains cxip_mr_lac_cache (breaks mr.h/ctrl.h cycle)
        "mr.h",            # Uses cxip_mr_lac_cache from mr_lac_cache.h
        "ctrl.h",          # Uses cxip_mr_lac_cache from mr_lac_cache.h
        "dom.h", "av.h", "fabric.h", "auth.h",
        "req.h", "fc.h", "msg_hpc.h", "rma.h", "atomic.h", "txc.h", "rxc.h",
        "curl.h", "repsum.h", "coll_trace.h", "coll.h", "zbcoll.h", "ep.h",
        "req_buf.h", "ptelist_buf.h", "rdzv_pte.h", "portals_table.h",
        "info.h", "nic.h", "telemetry.h",
    ]

    header_order = compute_header_order(dep_graph, fallback_order)
    print(f"Header include order: {header_order}", file=sys.stderr)

    # Get list of header names that exist
    existing_headers = {Path(h).name for h in headers.keys()}
    existing_headers.add("enums.h")  # Always include enums.h
    existing_headers.add("mr_lac_cache.h")  # Include cycle-breaking header

    # Include in computed order
    for h in header_order:
        if h in existing_headers:
            lines.append(f'#include "cxip/{h}"')

    # Add inline functions after all types are defined
    if inline_functions:
        lines.append("")
        lines.append("/*")
        lines.append(" * Inline function definitions")
        lines.append(" *")
        lines.append(" * These are kept here (not in split headers) because they often")
        lines.append(" * access struct members from multiple modules, requiring all types")
        lines.append(" * to be fully defined first.")
        lines.append(" */")
        lines.append("")

        # Sort by original source line to preserve order
        sorted_inlines = sorted(inline_functions, key=lambda s: s.start_line)
        for func in sorted_inlines:
            lines.append(func.text)
            lines.append("")

    lines.append("#endif /* _CXIP_PROV_H_ */")
    lines.append("")

    return '\n'.join(lines)


def main():
    cxip_h_path = Path("prov/cxi/include/cxip.h")
    output_dir = Path("prov/cxi/include/cxip")

    if not cxip_h_path.exists():
        print(f"Error: {cxip_h_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Check if we're receiving input from a pipeline
    if not sys.stdin.isatty():
        print("Reading refactor plan from stdin (pipeline mode)...", file=sys.stderr)
        plan = load_refactor_plan()
    else:
        plan_path = Path("prov/cxi/scripts/refactor_plan.json")
        if not plan_path.exists():
            print(f"Error: {plan_path} not found. Run generate_refactor_plan.py first.",
                  file=sys.stderr)
            print("Or pipe the output: ./generate_refactor_plan.py | ./apply_refactor.py",
                  file=sys.stderr)
            sys.exit(1)
        print("Loading refactor plan from file...", file=sys.stderr)
        plan = load_refactor_plan(str(plan_path))

    # Build a map of symbol name -> target header
    symbol_to_header: dict[str, str] = {}
    for header, info in plan["new_headers"].items():
        for func in info.get("functions", []):
            symbol_to_header[func] = header
        for typ in info.get("types", []):
            symbol_to_header[typ] = header
        for macro in info.get("macros", []):
            symbol_to_header[macro] = header

    print(f"Found {len(symbol_to_header)} symbols to extract", file=sys.stderr)

    # Read and parse cxip.h - use the backup if it exists (original content)
    backup_path = cxip_h_path.with_suffix('.h.orig')
    if backup_path.exists():
        print("Using backup cxip.h.orig for symbol extraction...", file=sys.stderr)
        content = backup_path.read_text()
    else:
        print("Parsing cxip.h...", file=sys.stderr)
        content = cxip_h_path.read_text()
    root = SgRoot(content, "c")

    # Collect symbol names by kind
    macro_names = {name for name, header in symbol_to_header.items()
                   if any(name in info.get("macros", [])
                         for info in plan["new_headers"].values())}
    type_names = {name for name, header in symbol_to_header.items()
                  if any(name in info.get("types", [])
                        for info in plan["new_headers"].values())}
    func_names = {name for name, header in symbol_to_header.items()
                  if any(name in info.get("functions", [])
                        for info in plan["new_headers"].values())}

    print(f"Looking for: {len(macro_names)} macros, {len(type_names)} types, "
          f"{len(func_names)} functions", file=sys.stderr)

    # Extract symbols
    print("Extracting macros...", file=sys.stderr)
    extracted_macros = extract_macro_definitions(content, macro_names)
    print(f"  Found {len(extracted_macros)} macros", file=sys.stderr)

    print("Extracting types...", file=sys.stderr)
    extracted_enums, extracted_types = extract_type_definitions(root, content, type_names)
    print(f"  Found {len(extracted_enums)} enums (-> enums.h)", file=sys.stderr)
    print(f"  Found {len(extracted_types)} other types", file=sys.stderr)

    print("Extracting functions...", file=sys.stderr)
    extracted_functions, extracted_inlines = extract_function_declarations(root, content, func_names)
    print(f"  Found {len(extracted_functions)} non-inline functions", file=sys.stderr)
    print(f"  Found {len(extracted_inlines)} inline functions (kept in cxip.h)", file=sys.stderr)

    # Extract function pointer typedefs (not caught by ast-grep)
    # These are needed by structs that use them, even if not in the plan
    print("Extracting function pointer typedefs...", file=sys.stderr)
    func_ptr_typedefs = extract_function_pointer_typedefs(content)
    # Remove any already extracted
    for name in list(func_ptr_typedefs.keys()):
        if name in extracted_types:
            del func_ptr_typedefs[name]
    if func_ptr_typedefs:
        print(f"  Found {len(func_ptr_typedefs)} function pointer typedefs: {list(func_ptr_typedefs.keys())}",
              file=sys.stderr)
        # Add these to extracted_types - they'll go to common.h since they're not in the plan
        extracted_types.update(func_ptr_typedefs)

    # Combine all extracted symbols (excluding inline functions and enums - they have special handling)
    # IMPORTANT: In C, the same name can be used for both a struct/union/enum and a function
    # (e.g., struct cxip_domain and int cxip_domain(...)). We use composite keys to avoid collisions.
    all_extracted = {}
    for name, sym in extracted_macros.items():
        all_extracted[f"{name}:macro"] = sym
    for name, sym in extracted_types.items():
        all_extracted[f"{name}:type"] = sym
    for name, sym in extracted_functions.items():
        all_extracted[f"{name}:function"] = sym

    # Report symbols not found in cxip.h (they might be in .c files)
    # Note: inline functions and enums are tracked separately
    # Strip the :kind suffix for comparison with symbol_to_header
    extracted_names = {key.rsplit(':', 1)[0] for key in all_extracted.keys()}
    all_symbol_names = extracted_names | set(extracted_inlines.keys()) | set(extracted_enums.keys())
    not_found = set(symbol_to_header.keys()) - all_symbol_names
    if not_found:
        print(f"\nSymbols not found in cxip.h ({len(not_found)}):", file=sys.stderr)
        for name in sorted(not_found)[:20]:
            details = plan.get("symbol_details", {}).get(name, {})
            defined = details.get("defined_in", [])
            print(f"  {name}: defined in {defined}", file=sys.stderr)
        if len(not_found) > 20:
            print(f"  ... and {len(not_found) - 20} more", file=sys.stderr)

    # Group extracted symbols by target header (excluding inline functions)
    headers: dict[str, HeaderContent] = defaultdict(lambda: HeaderContent(path=""))
    for key, sym in all_extracted.items():
        # Extract the original symbol name from composite key
        name = key.rsplit(':', 1)[0]
        target_header = symbol_to_header.get(name)
        if target_header:
            if headers[target_header].path == "":
                headers[target_header].path = target_header
            headers[target_header].symbols.append(sym)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate enums.h first (contains all enum definitions)
    if extracted_enums:
        enums_header_path = output_dir / "enums.h"
        enums_content = generate_enums_header(list(extracted_enums.values()))
        enums_header_path.write_text(enums_content)
        print(f"\nCreated {enums_header_path} ({len(extracted_enums)} enums)", file=sys.stderr)

    # Generate mr_lac_cache.h to break the circular dependency between mr.h and ctrl.h
    # cxip_mr_lac_cache is embedded in ctrl.h but defined in mr.h
    mr_lac_cache_sym = extracted_types.pop("cxip_mr_lac_cache", None)
    if mr_lac_cache_sym:
        # Also remove from the mr.h header's symbols list
        mr_h_path = "prov/cxi/include/cxip/mr.h"
        if mr_h_path in headers:
            headers[mr_h_path].symbols = [
                s for s in headers[mr_h_path].symbols
                if s.name != "cxip_mr_lac_cache"
            ]

        mr_lac_cache_header_path = output_dir / "mr_lac_cache.h"
        mr_lac_cache_content = generate_mr_lac_cache_header(mr_lac_cache_sym)
        mr_lac_cache_header_path.write_text(mr_lac_cache_content)
        print(f"Created {mr_lac_cache_header_path} (breaks mr.h/ctrl.h cycle)", file=sys.stderr)

    # Build set of all type names for forward declaration analysis
    all_type_names = set(extracted_types.keys()) | set(extracted_enums.keys())

    # Set of enum names that are in enums.h (to skip in individual headers)
    enum_names_in_enums_h = set(extracted_enums.keys())

    # Build type_to_header mapping for dependency analysis
    type_to_header: dict[str, str] = {}
    for name, sym in extracted_types.items():
        target = symbol_to_header.get(name)
        if target:
            type_to_header[name] = target
    for name, sym in extracted_enums.items():
        # Enums are in enums.h
        type_to_header[name] = "enums.h"
    # cxip_mr_lac_cache is in its own header (to break mr.h/ctrl.h cycle)
    if mr_lac_cache_sym:
        type_to_header["cxip_mr_lac_cache"] = "mr_lac_cache.h"

    # Build macro_to_header mapping for dependency analysis
    macro_to_header: dict[str, str] = {}
    for name, sym in extracted_macros.items():
        target = symbol_to_header.get(name)
        if target:
            macro_to_header[name] = target
    all_macro_names = set(extracted_macros.keys())

    # Generate and write new header files
    print(f"\nGenerating {len(headers)} new headers...", file=sys.stderr)
    for header_path, header_content in sorted(headers.items()):
        if not header_content.symbols:
            continue

        # Get types defined in this specific header
        types_in_header = {s.name for s in header_content.symbols if s.kind == "type"}

        output_path = output_dir / Path(header_path).name
        content = generate_header_file(header_content, all_type_names, types_in_header, enum_names_in_enums_h)
        output_path.write_text(content)
        print(f"  Created {output_path} ({len(header_content.symbols)} symbols)",
              file=sys.stderr)

    # Collect function pointer typedefs that aren't assigned to any header
    unassigned_func_ptr_typedefs = [
        sym for name, sym in func_ptr_typedefs.items()
        if name not in symbol_to_header
    ]

    # Generate new wrapper cxip.h with inline functions at the end
    original_cxip_content = cxip_h_path.read_text()
    inline_func_list = list(extracted_inlines.values())
    new_cxip_h = generate_wrapper_cxip_h(headers, type_to_header, macro_to_header,
                                          all_type_names, all_macro_names,
                                          original_cxip_content, inline_func_list,
                                          unassigned_func_ptr_typedefs)

    # Save original backup and overwrite cxip.h directly
    backup_path = cxip_h_path.with_suffix('.h.orig')
    if not backup_path.exists():
        backup_path.write_text(original_cxip_content)
        print(f"\nBacked up original cxip.h to {backup_path}", file=sys.stderr)

    # Overwrite cxip.h directly
    cxip_h_path.write_text(new_cxip_h)
    print(f"Updated {cxip_h_path}", file=sys.stderr)

    # Report summary
    print(f"\nSummary:", file=sys.stderr)
    print(f"  {len(extracted_enums)} enums -> enums.h (included first)", file=sys.stderr)
    if mr_lac_cache_sym:
        print(f"  1 type (cxip_mr_lac_cache) -> mr_lac_cache.h (breaks cycle)", file=sys.stderr)
    print(f"  {len(extracted_macros)} macros -> split headers", file=sys.stderr)
    print(f"  {len(extracted_types)} other types -> split headers", file=sys.stderr)
    print(f"  {len(extracted_functions)} non-inline functions -> split headers", file=sys.stderr)
    print(f"  {len(extracted_inlines)} inline functions -> kept in cxip.h (after all types)", file=sys.stderr)

    print("\nDone! Run 'make' to build.", file=sys.stderr)


if __name__ == "__main__":
    main()
