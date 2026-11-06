# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

#[[.rst:
LibfabricPlatform
-----------------

Select platform sources, definitions, and architecture flags.

Reads CMAKE_SYSTEM_NAME, CMAKE_SYSTEM_PROCESSOR, and libfabric_SOURCE_DIR.
Sets LIBFABRIC_PLATFORM_* lists, platform/architecture booleans, and
HAVE_MEMHOOKS_SUPPORT. Include before dependency and feature checks.

Variable-scope guard: initializes this build directory once per configure.
]]
include_guard()

# Platform detection
set(LIBFABRIC_LINUX FALSE)
set(LIBFABRIC_MACOS FALSE)
set(LIBFABRIC_FREEBSD FALSE)
set(LIBFABRIC_WINDOWS FALSE)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LIBFABRIC_LINUX TRUE)
  message(STATUS "Platform: Linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(LIBFABRIC_MACOS TRUE)
  message(STATUS "Platform: macOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
  set(LIBFABRIC_FREEBSD TRUE)
  message(STATUS "Platform: FreeBSD")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LIBFABRIC_WINDOWS TRUE)
  message(STATUS "Platform: Windows")
else()
  message(
    FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}. "
                "libfabric only builds on Linux, macOS, FreeBSD, and Windows."
  )
endif()

# Architecture detection
set(LIBFABRIC_X86_64 FALSE)
set(LIBFABRIC_AARCH64 FALSE)
set(LIBFABRIC_PPC64 FALSE)
set(LIBFABRIC_RISCV64 FALSE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  set(LIBFABRIC_X86_64 TRUE)
  message(STATUS "Architecture: x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64|arm64|ARM64)")
  set(LIBFABRIC_AARCH64 TRUE)
  message(STATUS "Architecture: aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(ppc64|powerpc64)")
  set(LIBFABRIC_PPC64 TRUE)
  message(STATUS "Architecture: ppc64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  set(LIBFABRIC_RISCV64 TRUE)
  message(STATUS "Architecture: riscv64")
else()
  message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Platform-specific sources and headers

if(LIBFABRIC_LINUX)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES
       "${libfabric_SOURCE_DIR}/src/unix/osd.c"
       "${libfabric_SOURCE_DIR}/src/linux/osd.c"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${libfabric_SOURCE_DIR}/include/unix"
       "${libfabric_SOURCE_DIR}/include/linux"
  )
elseif(LIBFABRIC_MACOS)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES
       "${libfabric_SOURCE_DIR}/src/osx/osd.c"
       "${libfabric_SOURCE_DIR}/src/unix/osd.c"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${libfabric_SOURCE_DIR}/include/unix"
       "${libfabric_SOURCE_DIR}/include/osx"
  )
elseif(LIBFABRIC_FREEBSD)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES
       "${libfabric_SOURCE_DIR}/src/unix/osd.c"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${libfabric_SOURCE_DIR}/include/unix"
       "${libfabric_SOURCE_DIR}/include/freebsd"
  )
elseif(LIBFABRIC_WINDOWS)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES
       "${libfabric_SOURCE_DIR}/src/windows/osd.c"
  )
  # Windows compatibility headers (pthread.h, sys/uio.h, etc.)
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${libfabric_SOURCE_DIR}/include/windows"
  )
endif()

# Platform-specific compile definitions
set(LIBFABRIC_PLATFORM_DEFINITIONS "")

if(LIBFABRIC_LINUX OR LIBFABRIC_FREEBSD)
  list(APPEND LIBFABRIC_PLATFORM_DEFINITIONS _GNU_SOURCE __USE_XOPEN2K8)
endif()

if(LIBFABRIC_MACOS)
  list(APPEND LIBFABRIC_PLATFORM_DEFINITIONS _GNU_SOURCE)
endif()

if(LIBFABRIC_WINDOWS)
  # Windows-specific definitions (from original vcxproj) _WINSOCKAPI_= prevents
  # windows.h from including winsock.h (we use WinSock2.h instead which must be
  # included before windows.h)
  list(
    APPEND
    LIBFABRIC_PLATFORM_DEFINITIONS
    WIN32
    _WINSOCKAPI_=
    _CRT_SECURE_NO_WARNINGS
    _WINSOCK_DEPRECATED_NO_WARNINGS
  )
endif()

# Check for memhooks support (Linux only on x86_64, aarch64, riscv64)
set(HAVE_MEMHOOKS_SUPPORT FALSE)
if(LIBFABRIC_LINUX)
  if(LIBFABRIC_X86_64
     OR LIBFABRIC_AARCH64
     OR LIBFABRIC_RISCV64
  )
    set(HAVE_MEMHOOKS_SUPPORT TRUE)
  endif()
endif()
