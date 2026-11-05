/*
 * neuron_efa_bdf: print the PCI BDF of the EFA NIC physically wired to a given
 * Neuron core, by asking the Neuron runtime at runtime instead of a static
 * per-instance table.
 *
 * Usage:   neuron_efa_bdf <neuron_core_id>
 * Output:  the BDF on stdout, e.g. "0000:c9:00.0\n"
 * Exit:    0 on success; non-zero with a diagnostic on stderr otherwise.
 *
 * The runtime symbols are resolved with dlopen so the binary itself does not
 * link the runtime lib: a host without the runtime fails the dlopen cleanly and
 * the caller treats that as "cannot determine sidelink NIC" (fail closed),
 * never a silent guess.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int nrt_status_t;
#define NRT_SUCCESS 0

#define NRT_TENSOR_PLACEMENT_DEVICE 0
#define NRT_FRAMEWORK_TYPE_NO_FW 1

typedef void nrt_tensor_t;

typedef nrt_status_t (*nrt_init_fn)(int, const char *, const char *);
typedef nrt_status_t (*nrt_tensor_allocate_fn)(int, int, size_t, const char *,
                                               nrt_tensor_t **);
typedef void (*nrt_tensor_free_fn)(nrt_tensor_t **);
typedef void *(*nrt_tensor_get_va_fn)(const nrt_tensor_t *);
typedef nrt_status_t (*nrt_get_attached_efa_bdf_fn)(const void *, char *,
                                                    size_t *);

static void *load_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "neuron_efa_bdf: symbol %s not found: %s\n", name,
            dlerror());
    exit(2);
  }
  return sym;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <neuron_core_id>\n", argv[0]);
    return 1;
  }

  char *end = NULL;
  long nc_id = strtol(argv[1], &end, 10);
  if (*end != '\0' || nc_id < 0 || nc_id > INT32_MAX) {
    fprintf(stderr, "neuron_efa_bdf: invalid neuron_core_id '%s'\n", argv[1]);
    return 1;
  }

  void *lib = dlopen("libnrt.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) {
    fprintf(stderr, "neuron_efa_bdf: failed to load libnrt.so.1: %s\n",
            dlerror());
    return 3;
  }

  nrt_init_fn nrt_init = (nrt_init_fn)load_sym(lib, "nrt_init");
  nrt_tensor_allocate_fn nrt_tensor_allocate =
      (nrt_tensor_allocate_fn)load_sym(lib, "nrt_tensor_allocate");
  nrt_tensor_free_fn nrt_tensor_free =
      (nrt_tensor_free_fn)load_sym(lib, "nrt_tensor_free");
  nrt_tensor_get_va_fn nrt_tensor_get_va =
      (nrt_tensor_get_va_fn)load_sym(lib, "nrt_tensor_get_va");
  nrt_get_attached_efa_bdf_fn nrt_get_attached_efa_bdf =
      (nrt_get_attached_efa_bdf_fn)load_sym(lib, "nrt_get_attached_efa_bdf");

  nrt_status_t rc = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "1.0", "1.0");
  if (rc != NRT_SUCCESS) {
    fprintf(stderr, "neuron_efa_bdf: nrt_init failed with status %d\n", rc);
    return 4;
  }

  nrt_tensor_t *tensor = NULL;
  rc = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, (int)nc_id, 4096,
                           "efa_probe", &tensor);
  if (rc != NRT_SUCCESS) {
    fprintf(stderr,
            "neuron_efa_bdf: nrt_tensor_allocate(vnc=%ld) failed with status "
            "%d\n",
            nc_id, rc);
    return 5;
  }

  void *va = nrt_tensor_get_va(tensor);
  if (!va) {
    fprintf(stderr, "neuron_efa_bdf: nrt_tensor_get_va returned NULL\n");
    nrt_tensor_free(&tensor);
    return 6;
  }

  char bdf[64];
  size_t len = sizeof(bdf);
  rc = nrt_get_attached_efa_bdf(va, bdf, &len);
  nrt_tensor_free(&tensor);
  if (rc != NRT_SUCCESS) {
    fprintf(stderr,
            "neuron_efa_bdf: nrt_get_attached_efa_bdf failed with status %d\n",
            rc);
    return 7;
  }

  if (len == 0 || len > sizeof(bdf)) {
    fprintf(stderr,
            "neuron_efa_bdf: runtime returned implausible BDF length %zu\n",
            len);
    return 8;
  }

  fwrite(bdf, 1, strnlen(bdf, len), stdout);
  fputc('\n', stdout);
  return 0;
}
