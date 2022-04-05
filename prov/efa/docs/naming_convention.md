## Function/variable naming convention of EFA libfabric provider

This document describes the function/variable naming convention of
libfabric EFA provider.

### Variable naming convention

Global variables' names must start with "g_" prefix. For exmaple,
`g_device_list` is a global variable that is an array of `struct efa_device`.

### Function naming convention

A function perform an action to an object. Unlike human language
that typically puts action in front of object,
function name puts the object name before action. For example,
a function that open an EFA endpoint is named `efa_ep_open`,
not `open_efa_ep`.

Functions are organized by objects, which means functions
acts on same object are put in same file. For example,
function `efa_ep_open()` and `efa_ep_close()` are located
in `efa_ep.c`.

Typical objects include `efa_ep`, `efa_cq`, `efa_device`.

Typical actions include:

`construct`: initializes data members of a struct, but does not allocate
the memory for the input. Typically, a function named `xxx_construct()`
will have its 1st argument to be pointer of type `struct xxx`.
For example, the 1st argument function `efa_device_construct()`'s is
a pointer of `struct efa_device *`. The function initializes the data
member of a `efa_device` struct.

`destruct` works in the opposite direction of `contruct`. It releases
resources associated with data members of a struct, but does not release
the memory of the struct itself.

`open` allocate an object and initializes its data members. Typically,
a function named `xxx_open()` will have an argument that is a pointer
to pointer of type `struct xxx`. On sucess, the argument will be set
to be pointing to a pointer of type `struct xxx`. For example,
the 3rd argument `ep_fid` of function `efa_ep_open()` is of type `struct fid_ep ** `.
On success, `ep_fid` will be pointing to a newly created `struct fid_ep` object.

`close` works in the opposite direction of `open`. It releases all
resources of an object, then free the memory of the object. Typically,
a function named `xxx_close` has an argument that is a pointer to `struct xxx`.
For example, `efa_ep_close()` takes a pointer named `ep` of type `struct ep_fid *` as
input. It releases the resources of `ep`, then release the memory pointed by ep.

`alloc` has the same behavior as `open`. It allocate memory of an object, then
initializes its data members. `open` is used for larger object, like endpoint (ep)
and completion queue (cq). `alloc` is used on smaller object, like `tx_entry`
and `rx_entry`.

`release` works on the opposite direction of `alloc`, and is used on object
that `alloc` is used on.

`initialize` is used to initialze global variables. For example, `efa_device_list_initialize`
initializeds the global variable `g_device_list` and `g_device_cnt`.

`finalize` works in the opposite direction of `initialize`, which means it is used
to release resources associated with global variables.
