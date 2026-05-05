NVCC=nvcc
NVCC_FLAGS=-G -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100
CUDA_FLAGS=-Xcompiler -fPIC
EFA_DP_DIRECT ?=
prefix ?= /usr/local
libdir = $(prefix)/lib

all: libefagda.so

libefagda.so: efa_gda_kernels.o
	$(NVCC) $(NVCC_FLAGS) -shared -o libefagda.so efa_gda_kernels.o

efa_gda_kernels.o: efa_gda_kernels.cu efa_gda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c efa_gda_kernels.cu $(CUDA_FLAGS) -I$(EFA_DP_DIRECT)/CUDA/src

install: libefagda.so
	install -d $(libdir)
	install -m 755 libefagda.so $(libdir)

clean:
	rm -f *.o libefagda.so
