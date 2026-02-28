# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(CheckSymbolExists)

set(CUDAToolkit_HAS_DMABUF FALSE)
set(CUDAToolkit_HAS_DMABUF_PCIE FALSE)

if(CUDAToolkit_FOUND)
  set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})

  # Check for dmabuf handle support
  check_symbol_exists(
    cuMemGetHandleForAddressRange "cuda.h" _HAVE_CUDA_DMABUF_HANDLE
  )
  check_symbol_exists(
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED "cuda.h" _HAVE_CUDA_DMABUF_ATTR
  )
  check_symbol_exists(
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD "cuda.h" _HAVE_CUDA_DMABUF_FD
  )

  if(_HAVE_CUDA_DMABUF_HANDLE
     AND _HAVE_CUDA_DMABUF_ATTR
     AND _HAVE_CUDA_DMABUF_FD
  )
    set(CUDAToolkit_HAS_DMABUF TRUE)
  endif()

  # Check for PCIe BAR1 mapping type support
  check_symbol_exists(
    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE "cuda.h" _HAVE_CUDA_DMABUF_PCIE
  )
  if(_HAVE_CUDA_DMABUF_PCIE)
    set(CUDAToolkit_HAS_DMABUF_PCIE TRUE)
  endif()

  unset(CMAKE_REQUIRED_INCLUDES)
endif()
