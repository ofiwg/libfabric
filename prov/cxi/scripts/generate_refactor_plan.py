#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Generate a refactoring plan from symbol analysis.

Improved heuristics:
1. Use naming conventions first - cxip_foo_* belongs in foo.h
2. Detect callback functions - functions assigned to struct fields aren't dead
3. Handle fundamental types - widely-used types go to logical home based on name
4. Track type dependencies for proper header ordering
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from typing import Literal


@dataclass
class SymbolInfo:
    name: str
    kind: Literal["function", "type", "macro"]
    defined_in: list[str]
    declared_in: list[str]
    used_in: list[str]
    is_static: bool = False
    is_inline: bool = False
    is_definition: bool = False
    full_text: str = ""
    signature: str = ""


@dataclass
class RefactorPlan:
    symbol_locations: dict[str, str] = field(default_factory=dict)
    new_headers: dict[str, list[str]] = field(default_factory=dict)
    private_symbols: dict[str, list[str]] = field(default_factory=dict)
    inline_handling: dict[str, str] = field(default_factory=dict)
    likely_callbacks: list[str] = field(default_factory=list)


def is_src_file(path: str) -> bool:
    return "prov/cxi/src/" in path and path.endswith(".c")


def is_test_file(path: str) -> bool:
    return "prov/cxi/test/" in path


def is_header_file(path: str) -> bool:
    return path.endswith(".h")


def is_main_header(path: str) -> bool:
    return path.endswith("cxip.h") and "include/cxip.h" in path


def get_tu_name(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("cxip_"):
        return stem[5:]
    return stem


def get_header_for_tu(tu_name: str) -> str:
    return f"prov/cxi/include/cxip/{tu_name}.h"


# Naming convention patterns - order matters (more specific first)
NAMING_PATTERNS = [
    # Specific subsystem patterns
    (r'^cxip_txc_hpc', 'txc'),
    (r'^cxip_txc_rnr', 'txc'),
    (r'^cxip_rxc_hpc', 'rxc'),
    (r'^cxip_rxc_rnr', 'rxc'),
    (r'^cxip_msg_hpc', 'msg_hpc'),
    (r'^cxip_msg_rnr', 'msg_rnr'),
    (r'^cxip_rdzv_pte', 'rdzv_pte'),
    (r'^cxip_rdzv_match', 'rdzv_pte'),
    (r'^cxip_rdzv_nomatch', 'rdzv_pte'),
    (r'^cxip_ptelist_buf', 'ptelist_buf'),
    (r'^cxip_req_buf', 'req_buf'),
    (r'^cxip_coll_trace', 'coll_trace'),
    (r'^cxip_ep_obj', 'ep'),
    (r'^cxip_ep_zbcoll', 'zbcoll'),
    (r'^cxip_ep_coll', 'coll'),

    # General patterns
    (r'^cxip_txc_', 'txc'),
    (r'^cxip_rxc_', 'rxc'),
    (r'^cxip_mr_', 'mr'),
    (r'^cxip_cq_', 'cq'),
    (r'^cxip_eq_', 'eq'),
    (r'^cxip_ep_', 'ep'),
    (r'^cxip_av_', 'av'),
    (r'^cxip_cntr_', 'cntr'),
    (r'^cxip_domain_', 'dom'),
    (r'^cxip_dom_', 'dom'),
    (r'^cxip_pte_', 'pte'),
    (r'^cxip_cmdq_', 'cmdq'),
    (r'^cxip_evtq_', 'evtq'),
    (r'^cxip_ctrl_', 'ctrl'),
    (r'^cxip_coll_', 'coll'),
    (r'^cxip_zbcoll_', 'zbcoll'),
    (r'^cxip_curl_', 'curl'),
    (r'^cxip_rma_', 'rma'),
    (r'^cxip_atomic_', 'atomic'),
    (r'^cxip_msg_', 'msg'),
    (r'^cxip_if_', 'if'),
    (r'^cxip_iomm_', 'iomm'),
    (r'^cxip_fabric_', 'fabric'),
    (r'^cxip_telemetry_', 'telemetry'),
    (r'^cxip_nic_', 'nic'),
    (r'^cxip_info_', 'info'),
    (r'^cxip_rep_', 'repsum'),
    (r'^cxip_faults_', 'faults'),
    (r'^cxip_portals_table', 'portals_table'),
    (r'^cxip_lni', 'if'),
    (r'^cxip_recv_', 'msg'),
    (r'^cxip_send_', 'msg'),
    (r'^cxip_ux_', 'msg'),
    (r'^cxip_fc_', 'fc'),
    (r'^cxip_map', 'iomm'),
    (r'^cxip_unmap', 'iomm'),
    (r'^cxip_copy_', 'mr'),
    (r'^cxip_generic_', 'mr'),
    (r'^cxip_tree_', 'zbcoll'),
    (r'^cxip_check_auth', 'auth'),
    (r'^cxip_gen_auth', 'auth'),

    # Type-specific patterns (for struct names)
    (r'^cxip_txc$', 'txc'),
    (r'^cxip_rxc$', 'rxc'),
    (r'^cxip_ep$', 'ep'),
    (r'^cxip_mr$', 'mr'),
    (r'^cxip_cq$', 'cq'),
    (r'^cxip_eq$', 'eq'),
    (r'^cxip_cntr$', 'cntr'),
    (r'^cxip_av$', 'av'),
    (r'^cxip_domain$', 'dom'),
    (r'^cxip_fabric$', 'fabric'),
    (r'^cxip_pte$', 'pte'),
    (r'^cxip_cmdq$', 'cmdq'),
    (r'^cxip_evtq$', 'evtq'),
    (r'^cxip_req$', 'req'),
    (r'^cxip_md$', 'mr'),
    (r'^cxip_if$', 'if'),
    (r'^cxip_addr$', 'addr'),
    (r'^cxip_environment$', 'env'),
    (r'^cxip_env$', 'env'),

    # Additional type patterns to reduce common.h usage
    (r'^cxip_req_', 'req'),      # cxip_req_send, cxip_req_recv, etc.
    (r'^cxip_repsum$', 'repsum'),
    (r'^cxip_dbl_bits$', 'repsum'),
    (r'^_bits2dbl$', 'repsum'),
    (r'^_dbl2bits$', 'repsum'),
    (r'^_decompose_dbl$', 'repsum'),
    (r'^cxip_ctrl$', 'ctrl'),
    (r'^cxip_ptelist_req$', 'ptelist_buf'),
    (r'^cxip_fltval$', 'coll'),
    (r'^cxip_fltminmax$', 'coll'),
    (r'^cxip_intval$', 'coll'),
    (r'^cxip_iminmax$', 'coll'),
    (r'^curl_ops$', 'curl'),
    (r'^cxip_match_bits$', 'msg'),
    (r'^cxip_llring_mode$', 'cmdq'),
    (r'^cxip_le_type$', 'pte'),
    (r'^cxip_amo_req_type$', 'atomic'),
    (r'^cxip_ats_mlock_mode$', 'iomm'),
    (r'^cxip_fid_list$', 'cq'),
    (r'^cxip_remap_cp$', 'if'),
    (r'^def_event_ht$', 'evtq'),

    # Inline utility function patterns
    (r'^is_netsim$', 'ep'),
    (r'^cxip_txq_ring$', 'cmdq'),
    (r'^cxip_mac_to_nic$', 'if'),
    (r'^cxip_cacheline_size$', 'if'),
    (r'^cxip_adjust_remote_offset$', 'mr'),
    (r'^single_to_double_quote$', 'curl'),
    (r'^cxip_json_', 'curl'),
    (r'^cxip_set_env_', 'env'),
    (r'^cxip_set_recv_', 'rxc'),
    (r'^cxip_get_owner_srx$', 'rxc'),
    (r'^cxip_is_trig_req$', 'req'),
    (r'^cxip_no_discard$', 'msg'),
    (r'^cxip_software_pte_allowed$', 'pte'),
    (r'^cxip_stx_alloc$', 'txc'),
    (r'^fls64$', 'if'),
    (r'^cxi_tc_str$', 'if'),

    # Macro patterns for common macros
    (r'^CXIP_ADDR_', 'addr'),
    (r'^CXIP_TAG_', 'msg'),
    (r'^CXIP_ALIGN', 'common'),
    (r'^ARRAY_SIZE$', 'common'),
    (r'^CEILING$', 'common'),
    (r'^FLOOR$', 'common'),
    (r'^CXIP_DBG$', 'log'),
    (r'^CXIP_INFO$', 'log'),
    (r'^CXIP_WARN', 'log'),
    (r'^CXIP_LOG$', 'log'),
    (r'^CXIP_FATAL$', 'log'),
    (r'^TXC_', 'txc'),
    (r'^RXC_', 'rxc'),
    (r'^DOM_', 'dom'),
]

# Callback function patterns - functions that are likely used as callbacks
CALLBACK_PATTERNS = [
    r'_cb$',           # Ends with _cb
    r'_callback$',     # Ends with _callback
    r'_handler$',      # Ends with _handler
    r'_progress$',     # Progress functions
    r'_recv$',         # Receive callbacks
    r'_send$',         # Send callbacks
    r'_complete$',     # Completion callbacks
    r'_ops$',          # Operation tables
]


def infer_home_from_name(name: str) -> str | None:
    """Infer the home TU from the symbol name using patterns."""
    for pattern, tu in NAMING_PATTERNS:
        if re.match(pattern, name, re.IGNORECASE):
            return tu
    return None


def is_likely_callback(name: str, sym: SymbolInfo) -> bool:
    """Check if a function is likely a callback based on naming patterns."""
    if sym.kind != "function":
        return False

    for pattern in CALLBACK_PATTERNS:
        if re.search(pattern, name):
            return True

    # Also check if function signature suggests callback (returns int, has specific params)
    # This is a heuristic based on common callback patterns
    return False


def find_home_tu_by_definition(sym: SymbolInfo) -> str | None:
    """Find home TU based on where the symbol is defined."""
    src_definitions = [f for f in sym.defined_in if is_src_file(f)]
    if src_definitions:
        return get_tu_name(src_definitions[0])
    return None


def find_home_tu_by_usage(sym: SymbolInfo) -> str | None:
    """Find home TU based on usage patterns (fallback)."""
    src_users = [f for f in sym.used_in if is_src_file(f)]
    if not src_users:
        return None

    tu_counts = Counter(get_tu_name(f) for f in src_users)
    if tu_counts:
        return tu_counts.most_common(1)[0][0]
    return None


def load_analysis(path: str | None = None) -> dict:
    """Load analysis from file or stdin if in a pipeline."""
    if path:
        with open(path) as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def extract_symbols(analysis: dict) -> dict[str, SymbolInfo]:
    """Extract and aggregate symbol information from analysis.

    Note: C allows the same name for a struct/union/enum and a function
    (e.g., struct cxip_domain and int cxip_domain(...)). We use composite
    keys like "name:type" and "name:function" to track both.
    """
    symbols: dict[str, SymbolInfo] = {}

    for file_info in analysis["files"]:
        file_path = file_info["path"]

        for func in file_info.get("function_decls", []):
            name = func["name"]
            # Use composite key to allow same name as type
            func_key = f"{name}:function"
            if func_key not in symbols:
                symbols[func_key] = SymbolInfo(
                    name=name,
                    kind="function",
                    defined_in=[],
                    declared_in=[],
                    used_in=[],
                    is_static=func.get("is_static", False),
                    is_inline=func.get("is_inline", False),
                    signature=func.get("signature", ""),
                )

            sym = symbols[func_key]
            if func.get("is_definition", False):
                if file_path not in sym.defined_in:
                    sym.defined_in.append(file_path)
                sym.is_definition = True
                sym.is_static = func.get("is_static", False)
                sym.is_inline = func.get("is_inline", False)
                sym.signature = func.get("signature", sym.signature)
            else:
                if file_path not in sym.declared_in:
                    sym.declared_in.append(file_path)

        for typedef in file_info.get("type_defs", []):
            name = typedef["name"]
            # Use composite key to allow same name as function
            type_key = f"{name}:type"
            if type_key not in symbols:
                symbols[type_key] = SymbolInfo(
                    name=name,
                    kind="type",
                    defined_in=[],
                    declared_in=[],
                    used_in=[],
                    full_text=typedef.get("full_text", ""),
                )

            sym = symbols[type_key]
            if not typedef.get("is_forward_decl", False):
                if file_path not in sym.defined_in:
                    sym.defined_in.append(file_path)
                sym.full_text = typedef.get("full_text", "")

        for macro in file_info.get("macro_defs", []):
            name = macro["name"]
            # Macros don't share namespace with types/functions, but use key for consistency
            macro_key = f"{name}:macro"
            if macro_key not in symbols:
                symbols[macro_key] = SymbolInfo(
                    name=name,
                    kind="macro",
                    defined_in=[],
                    declared_in=[],
                    used_in=[],
                    full_text=macro.get("full_text", ""),
                )

            sym = symbols[macro_key]
            if file_path not in sym.defined_in:
                sym.defined_in.append(file_path)

        for usage in file_info.get("usages", []):
            name = usage["symbol_name"]
            # Try to find the symbol in any kind
            for key in [f"{name}:function", f"{name}:type", f"{name}:macro"]:
                if key in symbols:
                    if file_path not in symbols[key].used_in:
                        symbols[key].used_in.append(file_path)

    return symbols


def analyze_symbol_visibility(symbols: dict[str, SymbolInfo]) -> RefactorPlan:
    """Determine where each symbol should live using improved heuristics."""
    plan = RefactorPlan()

    for key, sym in symbols.items():
        # Use sym.name for naming-based heuristics, but key for storage
        # This allows both 'cxip_domain:type' and 'cxip_domain:function' to coexist
        name = sym.name
        # Skip symbols not from main cxip.h or src files
        from_main_header = any(is_main_header(f) for f in sym.defined_in + sym.declared_in)
        from_src = any(is_src_file(f) for f in sym.defined_in)

        if not from_main_header and not from_src:
            plan.symbol_locations[key] = "external"
            continue

        src_users = set(f for f in sym.used_in if is_src_file(f))
        test_users = set(f for f in sym.used_in if is_test_file(f))

        # Check if this is likely a callback function
        if is_likely_callback(name, sym):
            plan.likely_callbacks.append(key)

        # HEURISTIC 1: Static symbols stay private
        if sym.is_static and sym.defined_in:
            src_defs = [f for f in sym.defined_in if is_src_file(f)]
            if src_defs:
                plan.symbol_locations[key] = f"private:{src_defs[0]}"
                if src_defs[0] not in plan.private_symbols:
                    plan.private_symbols[src_defs[0]] = []
                plan.private_symbols[src_defs[0]].append(name)
                continue

        # HEURISTIC 2: Use naming convention first
        home_tu = infer_home_from_name(name)

        # HEURISTIC 3: For functions, prefer where they're defined
        if home_tu is None and sym.kind == "function":
            home_tu = find_home_tu_by_definition(sym)

        # HEURISTIC 4: For types/macros defined in header, use name-based
        # Don't fall back to usage-based for types - that leads to poor placement

        # Determine if symbol needs to be exported
        needs_export = (len(src_users) > 1 or
                       len(test_users) > 0 or
                       is_likely_callback(name, sym))

        # CRITICAL: If a type/macro is DEFINED in the main header, it must go to a header
        # Even if it's only used in one place - it's part of the public API
        if from_main_header and sym.kind in ("type", "macro"):
            needs_export = True

        # Even single-use non-static functions might be callbacks
        if sym.kind == "function" and not sym.is_static and len(src_users) <= 1:
            if is_likely_callback(name, sym):
                needs_export = True
            elif from_main_header:  # Declared in header = intended to be public
                needs_export = True

        if needs_export:
            if home_tu:
                header = get_header_for_tu(home_tu)
                plan.symbol_locations[key] = f"header:{header}"
                if header not in plan.new_headers:
                    plan.new_headers[header] = []
                plan.new_headers[header].append(name)
            else:
                # Can't determine - put in common.h for now
                plan.symbol_locations[key] = "header:prov/cxi/include/cxip/common.h"
                if "prov/cxi/include/cxip/common.h" not in plan.new_headers:
                    plan.new_headers["prov/cxi/include/cxip/common.h"] = []
                plan.new_headers["prov/cxi/include/cxip/common.h"].append(name)
        elif len(src_users) == 1:
            # Private to one file
            src_file = list(src_users)[0]
            plan.symbol_locations[key] = f"private:{src_file}"
            if src_file not in plan.private_symbols:
                plan.private_symbols[src_file] = []
            plan.private_symbols[src_file].append(name)
        elif len(src_users) == 0 and not test_users:
            # Check if it's a callback or has declaration in header
            if is_likely_callback(name, sym) or from_main_header:
                if home_tu:
                    header = get_header_for_tu(home_tu)
                    plan.symbol_locations[key] = f"header:{header}"
                    if header not in plan.new_headers:
                        plan.new_headers[header] = []
                    plan.new_headers[header].append(name)
                else:
                    plan.symbol_locations[key] = "header:prov/cxi/include/cxip/common.h"
                    if "prov/cxi/include/cxip/common.h" not in plan.new_headers:
                        plan.new_headers["prov/cxi/include/cxip/common.h"] = []
                    plan.new_headers["prov/cxi/include/cxip/common.h"].append(name)
            else:
                plan.symbol_locations[key] = "dead_code"
        else:
            plan.symbol_locations[key] = "unknown"

    # Analyze inline functions
    for key, sym in symbols.items():
        name = sym.name
        if sym.kind == "function" and sym.is_inline:
            src_users = set(f for f in sym.used_in if is_src_file(f))

            if len(src_users) == 0:
                if is_likely_callback(name, sym):
                    plan.inline_handling[key] = "keep_for_callback"
                else:
                    plan.inline_handling[key] = "possibly_dead"
            elif len(src_users) == 1:
                plan.inline_handling[key] = f"private:{list(src_users)[0]}"
            else:
                loc = plan.symbol_locations.get(key, "")
                if loc.startswith("header:"):
                    plan.inline_handling[key] = f"keep_inline:{loc.split(':', 1)[1]}"
                else:
                    plan.inline_handling[key] = "make_regular_function"

    return plan


def generate_report(symbols: dict[str, SymbolInfo], plan: RefactorPlan) -> dict:
    """Generate a structured report."""
    # Build a reverse lookup: name -> list of keys (to handle name collisions)
    name_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, sym in symbols.items():
        name_to_keys[sym.name].append(key)

    report = {
        "summary": {
            "total_symbols": len(symbols),
            "functions": sum(1 for s in symbols.values() if s.kind == "function"),
            "types": sum(1 for s in symbols.values() if s.kind == "type"),
            "macros": sum(1 for s in symbols.values() if s.kind == "macro"),
            "likely_callbacks": len(plan.likely_callbacks),
        },
        "new_headers": {},
        "private_symbols": {},
        "inline_functions": plan.inline_handling,
        "likely_callbacks": sorted(plan.likely_callbacks),
        "location_summary": defaultdict(int),
    }

    for key, loc in plan.symbol_locations.items():
        if loc.startswith("header:"):
            report["location_summary"]["needs_header"] += 1
        elif loc.startswith("private:"):
            report["location_summary"]["private"] += 1
        elif loc == "dead_code":
            report["location_summary"]["dead_code"] += 1
        elif loc == "external":
            report["location_summary"]["external"] += 1
        else:
            report["location_summary"]["other"] += 1

    for header, sym_names in sorted(plan.new_headers.items()):
        # Categorize symbols by kind - need to look up by name, handling collisions
        funcs = []
        types = []
        macros = []
        for name in sym_names:
            # Find all keys for this name and categorize
            for key in name_to_keys.get(name, []):
                sym = symbols[key]
                if sym.kind == "function" and name not in funcs:
                    funcs.append(name)
                elif sym.kind == "type" and name not in types:
                    types.append(name)
                elif sym.kind == "macro" and name not in macros:
                    macros.append(name)

        report["new_headers"][header] = {
            "count": len(sym_names),
            "functions": sorted(set(funcs)),
            "types": sorted(set(types)),
            "macros": sorted(set(macros)),
        }

    for file, syms in sorted(plan.private_symbols.items()):
        report["private_symbols"][file] = {
            "count": len(syms),
            "symbols": sorted(syms),
        }

    report["location_summary"] = dict(report["location_summary"])

    return report


def main():
    # Check if we're receiving input from a pipeline
    if not sys.stdin.isatty():
        print("Reading symbol analysis from stdin (pipeline mode)...", file=sys.stderr)
        analysis = load_analysis()
    else:
        analysis_path = Path("prov/cxi/scripts/symbol_analysis.json")
        if not analysis_path.exists():
            print(f"Error: {analysis_path} not found. Run analyze_symbols.py first.",
                  file=sys.stderr)
            print("Or pipe the output: ./analyze_symbols.py | ./generate_refactor_plan.py",
                  file=sys.stderr)
            sys.exit(1)
        print("Loading symbol analysis from file...", file=sys.stderr)
        analysis = load_analysis(str(analysis_path))

    print("Extracting symbols...", file=sys.stderr)
    symbols = extract_symbols(analysis)
    print(f"Found {len(symbols)} unique symbols", file=sys.stderr)

    print("Analyzing symbol visibility with improved heuristics...", file=sys.stderr)
    plan = analyze_symbol_visibility(symbols)

    print("Generating report...", file=sys.stderr)
    report = generate_report(symbols, plan)

    # Add detailed symbol info - use sym.name as key in output for readability
    # but track collisions (same name with different kinds)
    report["symbol_details"] = {}
    for key, sym in symbols.items():
        name = sym.name
        # If there's already an entry for this name, append the kind to distinguish
        output_key = name if name not in report["symbol_details"] else key
        report["symbol_details"][output_key] = {
            "kind": sym.kind,
            "defined_in": sym.defined_in,
            "declared_in": sym.declared_in,
            "used_in_count": len(sym.used_in),
            "used_in_src": [f for f in sym.used_in if is_src_file(f)],
            "used_in_test": [f for f in sym.used_in if is_test_file(f)],
            "is_static": sym.is_static,
            "is_inline": sym.is_inline,
            "is_likely_callback": is_likely_callback(name, sym),
            "inferred_home": infer_home_from_name(name),
            "recommended_location": plan.symbol_locations.get(key, "unknown"),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
