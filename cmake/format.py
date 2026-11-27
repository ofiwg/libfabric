#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
"""Check maintained, Git-tracked CMake files; use --in-place to format.

Validate every project function/macro signature and reject unknown commands
before formatting. Generated files and untracked build trees are outside the
Git index; vendored GoogleTest is explicitly excluded.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys

try:
    import yaml
    from cmakelang import configuration, lex, parse
    from cmakelang.parse.statement_node import StatementNode
except ImportError:
    sys.exit(
        'Install the formatter with: '
        'python -m pip install -r cmake/requirements.txt'
    )


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / ".cmake-format.yaml"
# These are supplied by CMake, rather than declarations in this repository.
EXTERNAL_COMMANDS = {
    "check_required_components", "endforeach", "target_link_options"
}
EXCLUDED = ("prov/efa/test/gtest/",)
ARGUMENT_TYPES = {
    lex.TokenType.WORD,
    lex.TokenType.NUMBER,
    lex.TokenType.QUOTED_LITERAL,
    lex.TokenType.UNQUOTED_LITERAL,
    lex.TokenType.DEREF,
    lex.TokenType.BRACKET_ARGUMENT,
}


def project_files():
    """Include maintained CMake files; leave vendored upstream code alone."""
    paths = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    return [
        ROOT / path for path in paths
        if not path.startswith(EXCLUDED)
        and (Path(path).name == "CMakeLists.txt"
             or path.endswith((".cmake", ".cmake.in")))
        and (ROOT / path).is_file()
    ]


def statements(node):
    if isinstance(node, StatementNode):
        yield node
    for child in getattr(node, "children", ()):
        yield from statements(child)


def arguments(statement):
    return [
        token.spelling[1:-1] if token.type == lex.TokenType.QUOTED_LITERAL
        else token.spelling
        for token in statement.argtree.get_tokens()
        if token.type in ARGUMENT_TYPES
    ]


def signature(declaration, body):
    """Derive the registry entry from formal and parsed keyword arguments."""
    count = len(arguments(declaration)) - 1
    keywords = [node for node in body
                if node.get_funname() == "cmake_parse_arguments"]
    if len(keywords) > 1:
        raise ValueError(
            "multiple cmake_parse_arguments calls need a signature review"
        )
    result = {"pargs": count, "flags": set(), "kwargs": {}}
    if keywords:
        # cmake-format 0.6.13 needs a variadic positional group for keyword-only
        # commands; a zero-sized group cannot parse their leading keywords.
        result["pargs"] = count if count else "*"
        args = arguments(keywords[0])
        offset = 3 if args[0] == "PARSE_ARGV" else 1
        if len(args) < offset + 3:
            raise ValueError("incomplete cmake_parse_arguments signature")
        flags, single, multiple = args[offset:offset + 3]
        if any("$" in value or "[" in value
               for value in (flags, single, multiple)):
            raise ValueError("keyword lists must be literals for signature checking")
        names = [name for group in (flags, single, multiple)
                 for name in group.split(";") if name]
        if len(names) != len(set(names)):
            raise ValueError("duplicate cmake_parse_arguments keyword or flag")
        result["flags"] = set(filter(None, flags.split(";")))
        result["kwargs"] = {name: 1 for name in single.split(";") if name}
        result["kwargs"].update({name: "*" for name in multiple.split(";") if name})
    elif any(re.search(r"\$\{ARG[VN](?:\}|[0-9])", token.spelling)
             for node in body for token in node.argtree.get_tokens()):
        if count > 1:
            raise ValueError(
                "variadic functions with multiple fixed arguments "
                "need a signature review"
            )
        result["pargs"] = "+" if count else "*"
    return result


def check_registry(paths, options):
    registry = options["parse"]["additional_commands"]
    config = configuration.Configuration(**deepcopy(options))
    database = parse.funs.get_parse_db()
    database.update(parse.funs.get_funtree(config.parse.fn_spec))
    declarations = {}
    used = {}
    errors = []
    sources = {}
    for path in paths:
        sources[path] = path.read_text(encoding="utf-8")
        # Extract declarations independently of the registry being checked.
        # A stale arity must not cause its own parser to fail before diagnosis.
        tree = parse.parse(lex.tokenize(sources[path]))
        stack = []
        for node in statements(tree):
            name = node.get_funname()
            location = f"{path.relative_to(ROOT)}:{node.funnode.token.begin.line}"
            used.setdefault(name, location)
            if name in ("function", "macro"):
                stack.append((node, [], location))
            elif name in ("endfunction", "endmacro"):
                declaration, body, origin = stack.pop()
                custom = arguments(declaration)[0].lower()
                if custom in declarations:
                    errors.append(f"{origin}: duplicate declaration for {custom}")
                try:
                    declarations[custom] = signature(declaration, body)
                except ValueError as error:
                    errors.append(f"{origin}: {custom}: {error}")
            elif stack:
                stack[-1][1].append(node)
    for name in sorted(declarations.keys() - registry.keys()):
        errors.append(f"Missing custom command registration: {name}")
    for name in sorted(registry.keys() - declarations.keys() - EXTERNAL_COMMANDS):
        errors.append(f"Registration has no project declaration: {name}")
    for name in sorted(declarations.keys() & registry.keys()):
        actual = registry[name]
        actual = {"pargs": actual.get("pargs", "*"),
                  "flags": set(actual.get("flags", [])),
                  "kwargs": actual.get("kwargs", {})}
        if declarations[name] != actual:
            errors.append(
                f"Signature mismatch for {name}: "
                f"expected {declarations[name]}, got {actual}"
            )
    for name in sorted(used.keys() - database.keys()):
        errors.append(f"{used[name]}: command missing from formatter parser: {name}")
    if errors:
        raise ValueError("\n".join(errors))
    # Now exercise every call with the same parser database as the formatter.
    for path, source in sources.items():
        try:
            parse.parse(lex.tokenize(source),
                        parse.ParseContext(database, config=config))
        except Exception as error:
            raise ValueError(
                f"{path.relative_to(ROOT)}: configured formatter parse failed: "
                f"{type(error).__name__}: {error}"
            ) from error
    print(
        f"Checked {len(declarations)} custom command signatures "
        f"and parsed {len(paths)} CMake files."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="check formatting (default)")
    mode.add_argument("--in-place", action="store_true", help="apply formatting")
    mode.add_argument("--registry-only", action="store_true",
                      help="check signatures and parsing only")
    args = parser.parse_args()
    paths = project_files()
    try:
        check_registry(paths, yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    except ValueError as error:
        sys.exit(str(error))
    if not args.registry_only:
        return subprocess.call(
            [sys.executable, "-m", "cmakelang.format",
             "--in-place" if args.in_place else "--check",
             "-c", str(CONFIG), "--", *map(str, paths)], cwd=ROOT
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
