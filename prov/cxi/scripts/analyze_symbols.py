#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "ast-grep-py>=0.31.0",
# ]
# ///
"""
Analyze CXI provider source code to extract symbol information.

This script extracts:
- Function declarations and definitions
- Type definitions (structs, enums, typedefs, unions)
- Macro definitions
- Usage sites for all of the above

Output is a JSON report that can be used to plan header refactoring.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Literal
from ast_grep_py import SgRoot, SgNode


@dataclass
class Location:
    file: str
    line: int
    column: int


@dataclass
class FunctionDecl:
    name: str
    location: Location
    is_static: bool
    is_inline: bool
    is_definition: bool  # True if this is a definition, False if just declaration
    signature: str  # Full function signature for matching


@dataclass
class TypeDef:
    name: str
    location: Location
    kind: Literal["struct", "enum", "union", "typedef"]
    is_forward_decl: bool
    full_text: str  # For complex types


@dataclass
class MacroDef:
    name: str
    location: Location
    is_function_like: bool
    full_text: str


@dataclass
class SymbolUsage:
    symbol_name: str
    location: Location
    usage_kind: Literal["call", "type_ref", "macro_ref", "pointer_only"]
    # pointer_only means we only use a pointer to this type, so forward decl suffices


@dataclass
class FileAnalysis:
    path: str
    function_decls: list[FunctionDecl] = field(default_factory=list)
    type_defs: list[TypeDef] = field(default_factory=list)
    macro_defs: list[MacroDef] = field(default_factory=list)
    usages: list[SymbolUsage] = field(default_factory=list)


def get_location(node: SgNode, file_path: str) -> Location:
    """Extract location from an ast-grep node."""
    range_info = node.range()
    return Location(
        file=file_path,
        line=range_info.start.line + 1,  # ast-grep uses 0-indexed lines
        column=range_info.start.column,
    )


def find_child_by_kind(node: SgNode, kind: str) -> SgNode | None:
    """Find first child with given kind."""
    for child in node.children():
        if child.kind() == kind:
            return child
    return None


def find_all_children_by_kind(node: SgNode, kind: str) -> list[SgNode]:
    """Find all children with given kind."""
    return [child for child in node.children() if child.kind() == kind]


def find_identifier_in_declarator(node: SgNode) -> str | None:
    """Recursively find the identifier in a declarator."""
    if node.kind() == "identifier":
        return node.text()

    # Check field access first if available
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


def analyze_functions(root: SgRoot, file_path: str) -> list[FunctionDecl]:
    """Extract function declarations and definitions."""
    functions = []

    # Find function definitions
    for node in root.root().find_all(kind="function_definition"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        func_name = find_identifier_in_declarator(declarator)
        if func_name is None:
            continue

        full_text = node.text()

        # Check for static/inline by looking at storage_class_specifier children
        # or by checking the text before the function name
        prefix = full_text.split(func_name)[0] if func_name in full_text else ""
        is_static = "static" in prefix
        is_inline = "inline" in prefix

        # Get signature (everything before the body)
        sig_end = full_text.find('{')
        signature = full_text[:sig_end].strip() if sig_end > 0 else full_text

        functions.append(FunctionDecl(
            name=func_name,
            location=get_location(node, file_path),
            is_static=is_static,
            is_inline=is_inline,
            is_definition=True,
            signature=signature,
        ))

    # Find function declarations (no body)
    for node in root.root().find_all(kind="declaration"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        # Check if this has a function_declarator somewhere
        has_func_decl = False
        check_node = declarator
        while check_node:
            if check_node.kind() == "function_declarator":
                has_func_decl = True
                break
            check_node = find_child_by_kind(check_node, "function_declarator")
            if check_node is None:
                # Also check pointer_declarator
                ptr = find_child_by_kind(declarator, "pointer_declarator")
                if ptr:
                    check_node = find_child_by_kind(ptr, "function_declarator")
                break

        if not has_func_decl:
            continue

        func_name = find_identifier_in_declarator(declarator)
        if func_name is None:
            continue

        full_text = node.text()
        prefix = full_text.split(func_name)[0] if func_name in full_text else ""
        is_static = "static" in prefix
        is_inline = "inline" in prefix

        functions.append(FunctionDecl(
            name=func_name,
            location=get_location(node, file_path),
            is_static=is_static,
            is_inline=is_inline,
            is_definition=False,
            signature=full_text.rstrip(';'),
        ))

    return functions


def analyze_types(root: SgRoot, file_path: str) -> list[TypeDef]:
    """Extract type definitions (struct, enum, union, typedef)."""
    types = []

    # Find struct/union/enum definitions
    for kind, type_kind in [("struct_specifier", "struct"),
                            ("union_specifier", "union"),
                            ("enum_specifier", "enum")]:
        for node in root.root().find_all(kind=kind):
            # Get the name (type_identifier child)
            name_node = find_child_by_kind(node, "type_identifier")
            if name_node is None:
                continue

            type_name = name_node.text()
            full_text = node.text()

            # Check if this is a forward declaration (no field_declaration_list)
            body = find_child_by_kind(node, "field_declaration_list")
            if body is None:
                body = find_child_by_kind(node, "enumerator_list")
            is_forward = body is None

            types.append(TypeDef(
                name=type_name,
                location=get_location(node, file_path),
                kind=type_kind,
                is_forward_decl=is_forward,
                full_text=full_text,
            ))

    # Find typedefs
    for node in root.root().find_all(kind="type_definition"):
        declarator = node.field("declarator")
        if declarator is None:
            continue

        # Get the typedef'd name - look for type_identifier
        type_name = None
        if declarator.kind() == "type_identifier":
            type_name = declarator.text()
        else:
            # Look in children
            for child in declarator.children():
                if child.kind() == "type_identifier":
                    type_name = child.text()
                    break
            # Also check if declarator itself contains the name
            if type_name is None:
                ti = find_child_by_kind(declarator, "type_identifier")
                if ti:
                    type_name = ti.text()

        if type_name is None:
            # Last resort: look for any identifier
            for child in declarator.children():
                if child.kind() == "identifier":
                    type_name = child.text()
                    break

        if type_name is None:
            continue

        types.append(TypeDef(
            name=type_name,
            location=get_location(node, file_path),
            kind="typedef",
            is_forward_decl=False,
            full_text=node.text(),
        ))

    return types


def analyze_macros(root: SgRoot, file_path: str) -> list[MacroDef]:
    """Extract macro definitions."""
    macros = []

    for node in root.root().find_all(kind="preproc_def"):
        name_node = node.field("name")
        if name_node is None:
            continue

        macro_name = name_node.text()
        full_text = node.text()

        macros.append(MacroDef(
            name=macro_name,
            location=get_location(node, file_path),
            is_function_like=False,
            full_text=full_text,
        ))

    for node in root.root().find_all(kind="preproc_function_def"):
        name_node = node.field("name")
        if name_node is None:
            continue

        macro_name = name_node.text()
        full_text = node.text()

        macros.append(MacroDef(
            name=macro_name,
            location=get_location(node, file_path),
            is_function_like=True,
            full_text=full_text,
        ))

    return macros


def analyze_usages(root: SgRoot, file_path: str, known_functions: set[str],
                   known_types: set[str], known_macros: set[str]) -> list[SymbolUsage]:
    """Find usages of known symbols."""
    usages = []
    seen = set()  # Avoid duplicates at same location

    # Find function calls
    for node in root.root().find_all(kind="call_expression"):
        func_node = node.field("function")
        if func_node is None:
            continue

        # Handle direct calls
        if func_node.kind() == "identifier":
            func_name = func_node.text()
            if func_name in known_functions:
                loc = get_location(node, file_path)
                key = (func_name, loc.line, loc.column, "call")
                if key not in seen:
                    seen.add(key)
                    usages.append(SymbolUsage(
                        symbol_name=func_name,
                        location=loc,
                        usage_kind="call",
                    ))

    # Find type references
    for node in root.root().find_all(kind="type_identifier"):
        type_name = node.text()
        if type_name in known_types:
            # Determine if this is pointer-only usage
            parent = node.parent()
            is_pointer_only = False
            if parent:
                gp = parent.parent()
                if gp and "pointer" in gp.kind():
                    is_pointer_only = True

            loc = get_location(node, file_path)
            kind = "pointer_only" if is_pointer_only else "type_ref"
            key = (type_name, loc.line, loc.column, kind)
            if key not in seen:
                seen.add(key)
                usages.append(SymbolUsage(
                    symbol_name=type_name,
                    location=loc,
                    usage_kind=kind,
                ))

    # Find struct/union/enum references (when used as types)
    for kind in ["struct_specifier", "union_specifier", "enum_specifier"]:
        for node in root.root().find_all(kind=kind):
            name_node = find_child_by_kind(node, "type_identifier")
            body = find_child_by_kind(node, "field_declaration_list")
            if body is None:
                body = find_child_by_kind(node, "enumerator_list")

            # Only count as usage if no body (reference, not definition)
            if name_node and body is None:
                type_name = name_node.text()
                if type_name in known_types:
                    parent = node.parent()
                    is_pointer_only = parent and "pointer" in parent.kind()

                    loc = get_location(node, file_path)
                    usage_kind = "pointer_only" if is_pointer_only else "type_ref"
                    key = (type_name, loc.line, loc.column, usage_kind)
                    if key not in seen:
                        seen.add(key)
                        usages.append(SymbolUsage(
                            symbol_name=type_name,
                            location=loc,
                            usage_kind=usage_kind,
                        ))

    # Find macro usages
    for node in root.root().find_all(kind="identifier"):
        ident = node.text()
        if ident in known_macros:
            # Make sure this isn't the macro definition itself
            parent = node.parent()
            if parent and parent.kind() in ("preproc_def", "preproc_function_def"):
                # Check if this is the name field
                name_field = parent.field("name")
                if name_field and name_field.text() == ident:
                    continue

            loc = get_location(node, file_path)
            key = (ident, loc.line, loc.column, "macro_ref")
            if key not in seen:
                seen.add(key)
                usages.append(SymbolUsage(
                    symbol_name=ident,
                    location=loc,
                    usage_kind="macro_ref",
                ))

    return usages


def analyze_file(file_path: Path) -> FileAnalysis:
    """Analyze a single C source file."""
    content = file_path.read_text()

    try:
        root = SgRoot(content, "c")
    except Exception as e:
        print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
        return FileAnalysis(path=str(file_path))

    analysis = FileAnalysis(path=str(file_path))
    analysis.function_decls = analyze_functions(root, str(file_path))
    analysis.type_defs = analyze_types(root, str(file_path))
    analysis.macro_defs = analyze_macros(root, str(file_path))

    return analysis


def main():
    # Find all C source and header files in prov/cxi
    cxi_dir = Path("prov/cxi")

    if not cxi_dir.exists():
        print(f"Error: {cxi_dir} does not exist. Run from libfabric root.",
              file=sys.stderr)
        sys.exit(1)

    # Note: This script outputs to stdout, which can be piped to generate_refactor_plan.py

    c_files = list(cxi_dir.rglob("*.c"))
    h_files = list(cxi_dir.rglob("*.h"))
    all_files = c_files + h_files

    print(f"Found {len(c_files)} C files and {len(h_files)} header files", file=sys.stderr)

    # First pass: collect all definitions
    all_analyses: list[FileAnalysis] = []
    known_functions: set[str] = set()
    known_types: set[str] = set()
    known_macros: set[str] = set()

    for file_path in all_files:
        print(f"Analyzing {file_path}...", file=sys.stderr)
        analysis = analyze_file(file_path)
        all_analyses.append(analysis)

        for func in analysis.function_decls:
            known_functions.add(func.name)
        for typedef in analysis.type_defs:
            known_types.add(typedef.name)
        for macro in analysis.macro_defs:
            known_macros.add(macro.name)

    print(f"Found {len(known_functions)} functions, {len(known_types)} types, "
          f"{len(known_macros)} macros", file=sys.stderr)

    # Second pass: find usages
    for file_path, analysis in zip(all_files, all_analyses):
        content = file_path.read_text()
        try:
            root = SgRoot(content, "c")
            analysis.usages = analyze_usages(root, str(file_path),
                                            known_functions, known_types, known_macros)
        except Exception as e:
            print(f"Warning: Failed to analyze usages in {file_path}: {e}", file=sys.stderr)

    # Convert to JSON-serializable format
    result = {
        "files": [asdict(a) for a in all_analyses],
        "summary": {
            "total_functions": len(known_functions),
            "total_types": len(known_types),
            "total_macros": len(known_macros),
            "files_analyzed": len(all_files),
        }
    }

    # Output JSON
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
