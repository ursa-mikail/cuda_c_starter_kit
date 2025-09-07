CC=nvcc
CFLAGS=-O3 -Xcompiler -fopenmp
TARGET=matmul_compare
SOURCE=matmul_compare.cu

# Check CUDA compiler
CUDA_CHECK := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_CHECK),)
    $(error CUDA compiler (nvcc) not found. Please install CUDA toolkit)
endif

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling CUDA matrix multiplication demo..."
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)
	@echo "Build successful!"

run: $(TARGET)
	@echo "Running demo with default N=1024..."
	./$(TARGET)

test-small: $(TARGET)
	@echo "Running test with 64x64 matrices..."
	./$(TARGET) 64

test-medium: $(TARGET)
	@echo "Running test with 256x256 matrices..."
	./$(TARGET) 256

test-large: $(TARGET)
	@echo "Running test with 1024x1024 matrices..."
	./$(TARGET) 1024

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)

gpu-info:
	@echo "CUDA GPU Information:"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not found"

check-deps:
	@echo "Checking dependencies..."
	@echo -n "CUDA compiler: "
	@which nvcc >/dev/null 2>&1 && echo "✓ Found" || echo "✗ Not found"
	@echo -n "OpenMP: "
	@echo '#include <omp.h>' | gcc -fopenmp -x c - -o /tmp/omp_test 2>/dev/null && echo "✓ Found" || echo "✗ Not found"
	@rm -f /tmp/omp_test

help:
	@echo "Available targets:"
	@echo "  all         - Build demo"
	@echo "  run         - Run default (N=1024)"
	@echo "  test-small  - Run 64x64 matrices"
	@echo "  test-medium - Run 256x256 matrices"
	@echo "  test-large  - Run 1024x1024 matrices"
	@echo "  clean       - Remove built files"
	@echo "  gpu-info    - Show GPU info"
	@echo "  check-deps  - Check CUDA & OpenMP"

.PHONY: all run test-small test-medium test-large clean gpu-info check-deps help
