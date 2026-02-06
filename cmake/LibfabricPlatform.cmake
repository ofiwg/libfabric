# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only

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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
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

# Platform-specific source files
set(LIBFABRIC_PLATFORM_SOURCES "")
set(LIBFABRIC_PLATFORM_HEADERS "")
set(LIBFABRIC_PLATFORM_INCLUDE_DIRS "")

if(LIBFABRIC_LINUX)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES "${CMAKE_SOURCE_DIR}/src/unix/osd.c"
       "${CMAKE_SOURCE_DIR}/src/linux/osd.c"
  )
  list(
    APPEND LIBFABRIC_PLATFORM_HEADERS "${CMAKE_SOURCE_DIR}/include/linux/osd.h"
    "${CMAKE_SOURCE_DIR}/include/linux/rdpmc.h"
    "${CMAKE_SOURCE_DIR}/include/unix/osd.h"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${CMAKE_SOURCE_DIR}/include/unix" "${CMAKE_SOURCE_DIR}/include/linux"
  )
elseif(LIBFABRIC_MACOS)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES "${CMAKE_SOURCE_DIR}/src/osx/osd.c"
       "${CMAKE_SOURCE_DIR}/src/unix/osd.c"
  )
  list(APPEND LIBFABRIC_PLATFORM_HEADERS
       "${CMAKE_SOURCE_DIR}/include/osx/osd.h"
       "${CMAKE_SOURCE_DIR}/include/unix/osd.h"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${CMAKE_SOURCE_DIR}/include/unix" "${CMAKE_SOURCE_DIR}/include/osx"
  )
elseif(LIBFABRIC_FREEBSD)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES "${CMAKE_SOURCE_DIR}/src/unix/osd.c")
  list(APPEND LIBFABRIC_PLATFORM_HEADERS
       "${CMAKE_SOURCE_DIR}/include/freebsd/osd.h"
       "${CMAKE_SOURCE_DIR}/include/unix/osd.h"
  )
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${CMAKE_SOURCE_DIR}/include/unix" "${CMAKE_SOURCE_DIR}/include/freebsd"
  )
elseif(LIBFABRIC_WINDOWS)
  list(APPEND LIBFABRIC_PLATFORM_SOURCES
       "${CMAKE_SOURCE_DIR}/src/windows/osd.c"
  )
  list(APPEND LIBFABRIC_PLATFORM_HEADERS
       "${CMAKE_SOURCE_DIR}/include/windows/osd.h"
  )
  # Windows compatibility headers (pthread.h, sys/uio.h, etc.)
  list(APPEND LIBFABRIC_PLATFORM_INCLUDE_DIRS
       "${CMAKE_SOURCE_DIR}/include/windows"
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
  list(APPEND LIBFABRIC_PLATFORM_DEFINITIONS WIN32 _WINSOCKAPI_=
       _CRT_SECURE_NO_WARNINGS _WINSOCK_DEPRECATED_NO_WARNINGS
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

# Export variables to parent scope if called from add_subdirectory
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(LIBFABRIC_LINUX
      ${LIBFABRIC_LINUX}
      PARENT_SCOPE
  )
  set(LIBFABRIC_MACOS
      ${LIBFABRIC_MACOS}
      PARENT_SCOPE
  )
  set(LIBFABRIC_FREEBSD
      ${LIBFABRIC_FREEBSD}
      PARENT_SCOPE
  )
  set(LIBFABRIC_WINDOWS
      ${LIBFABRIC_WINDOWS}
      PARENT_SCOPE
  )
  set(LIBFABRIC_X86_64
      ${LIBFABRIC_X86_64}
      PARENT_SCOPE
  )
  set(LIBFABRIC_AARCH64
      ${LIBFABRIC_AARCH64}
      PARENT_SCOPE
  )
  set(LIBFABRIC_PPC64
      ${LIBFABRIC_PPC64}
      PARENT_SCOPE
  )
  set(LIBFABRIC_RISCV64
      ${LIBFABRIC_RISCV64}
      PARENT_SCOPE
  )
  set(LIBFABRIC_PLATFORM_SOURCES
      ${LIBFABRIC_PLATFORM_SOURCES}
      PARENT_SCOPE
  )
  set(LIBFABRIC_PLATFORM_HEADERS
      ${LIBFABRIC_PLATFORM_HEADERS}
      PARENT_SCOPE
  )
  set(LIBFABRIC_PLATFORM_INCLUDE_DIRS
      ${LIBFABRIC_PLATFORM_INCLUDE_DIRS}
      PARENT_SCOPE
  )
  set(LIBFABRIC_PLATFORM_DEFINITIONS
      ${LIBFABRIC_PLATFORM_DEFINITIONS}
      PARENT_SCOPE
  )
  set(HAVE_MEMHOOKS_SUPPORT
      ${HAVE_MEMHOOKS_SUPPORT}
      PARENT_SCOPE
  )
endif()
