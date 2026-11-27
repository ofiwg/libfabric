# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# Configure CheckEfadv against a synthetic SDK missing one optional API.
cmake_minimum_required(VERSION 3.21)
project(EfadvFeatures LANGUAGES C)

set(_include "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${_include}/infiniband")
file(WRITE "${_include}/infiniband/verbs.h" "#pragma once\n")
set(_header "#pragma once\n")
foreach(_symbol IN ITEMS query_ah create_mem_comp_action destroy_comp_action
                         get_max_inline_data
)
  if(NOT MISSING STREQUAL _symbol)
    string(APPEND _header "static inline void efadv_${_symbol}(void) {}\n")
  endif()
endforeach()
if(MISSING STREQUAL "action_block_offset")
  string(APPEND _header "struct efadv_wq_attr { int unused; };\n")
else()
  string(
    APPEND
    _header
    "struct efadv_wq_attr { unsigned int comp_action_with_data_block_offset; };\n"
  )
endif()
file(WRITE "${_include}/infiniband/efadv.h" "${_header}")

add_library(Efadv::Efadv INTERFACE IMPORTED)
add_library(IBVerbs::IBVerbs INTERFACE IMPORTED)
set(Efadv_FOUND TRUE)
set(IBVerbs_FOUND TRUE)
set(Efadv_INCLUDE_DIRS "${_include}")
list(PREPEND CMAKE_MODULE_PATH "${MODULE_DIR}")
include(CheckEfadv)

if(NOT Efadv_FOUND)
  message(FATAL_ERROR "An optional feature disabled the EFA provider")
endif()
if(MISSING STREQUAL "NONE")
  if(NOT Efadv_HAVE_COMP_ACTION)
    message(FATAL_ERROR "Completion actions missing from a complete SDK")
  endif()
elseif(Efadv_HAVE_COMP_ACTION)
  message(FATAL_ERROR "Completion actions enabled without ${MISSING}")
endif()
