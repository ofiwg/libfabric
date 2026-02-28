# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
include(FindPackageHandleStandardArgs)
include(CheckCSourceCompiles)
include(CheckIncludeFile)

set(UFD_FOUND FALSE)
set(UFD_HAS_UNMAP FALSE)
set(UFD_HAS_THREAD_ID FALSE)

check_include_file("linux/userfaultfd.h" HAVE_USERFAULTFD_H)

if(HAVE_USERFAULTFD_H)
  # Check for __NR_userfaultfd
  check_c_source_compiles(
    "
    #include <sys/syscall.h>
    int main() {
      #ifndef __NR_userfaultfd
      #error no userfaultfd syscall
      #endif
      return 0;
    }
  "
    HAVE_NR_USERFAULTFD
  )

  if(HAVE_NR_USERFAULTFD)
    # Check for unmap support
    check_c_source_compiles(
      "
      #include <sys/types.h>
      #include <linux/userfaultfd.h>
      #include <unistd.h>
      #include <sys/syscall.h>
      #include <fcntl.h>
      #include <sys/ioctl.h>
      int main() {
        int fd;
        struct uffdio_api api_obj;
        api_obj.api = UFFD_API;
        api_obj.features = UFFD_FEATURE_EVENT_UNMAP |
                          UFFD_FEATURE_EVENT_REMOVE |
                          UFFD_FEATURE_EVENT_REMAP;
        fd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
        return ioctl(fd, UFFDIO_API, &api_obj);
      }
    "
      UFD_HAS_UNMAP
    )

    if(UFD_HAS_UNMAP)
      set(UFD_FOUND TRUE)

      # Check for thread id support
      check_c_source_compiles(
        "
        #include <sys/types.h>
        #include <linux/userfaultfd.h>
        #include <unistd.h>
        #include <sys/syscall.h>
        #include <fcntl.h>
        #include <sys/ioctl.h>
        int main() {
          int fd;
          struct uffdio_api api_obj;
          api_obj.api = UFFD_API;
          api_obj.features = UFFD_FEATURE_THREAD_ID |
                            UFFD_FEATURE_EVENT_UNMAP |
                            UFFD_FEATURE_EVENT_REMOVE |
                            UFFD_FEATURE_EVENT_REMAP;
          fd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
          return ioctl(fd, UFFDIO_API, &api_obj);
        }
      "
        UFD_HAS_THREAD_ID
      )
    endif()
  endif()
endif()

find_package_handle_standard_args(UFD REQUIRED_VARS UFD_FOUND)
