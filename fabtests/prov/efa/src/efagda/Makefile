CC=gcc
NVCC=nvcc
NVCC_FLAGS=-G -arch=sm_90
CFLAGS=-pthread
CUDA_FLAGS=-Xcompiler -fPIC
CUDA_INCLUDES=-I/usr/local/cuda/include
CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudart
prefix ?= /usr/local
libdir = $(prefix)/lib

all: libefagda.so

libefagda.so: cuda_kernel.o efagda.o
	$(NVCC) $(NVCC_FLAGS) -shared -o libefagda.so cuda_kernel.o efagda.o

cuda_kernel.o: cuda_kernel.cu cuda_kernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c cuda_kernel.cu $(CUDA_FLAGS)

efagda.o: efagda.cpp efagda.h cuda_kernel.cuh
	$(NVCC) $(NVCC_FLAGS) -c efagda.cpp $(CUDA_FLAGS)

install: libefagda.so
	install -d $(libdir)
	install -m 755 libefagda.so $(libdir)

clean:
	rm -f *.o $(libdir)/libefagda.so myapp
