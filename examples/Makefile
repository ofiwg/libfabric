CC		= gcc
INCLUDES	= -I$(LIBFABRIC_PREFIX)/include
LIBS		= -L$(LIBFABRIC_PREFIX)/lib
CFLAGS		= -g -O0 $(INCLUDES) $(LIBS)
LDFLAGS		= -lfabric
EXEC_PREFIX	= fi_example_

PROGRAMS = rdm msg rdm_tagged rdm_rma tcp_socket

EXECUTABLES = $(addprefix $(EXEC_PREFIX), $(PROGRAMS))

.PHONY: all
all: $(EXECUTABLES)

$(EXEC_PREFIX)%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXECUTABLES)
