CC=nvcc
CFLAGS=-O3 -Xcompiler -fopenmp
LIBS=-lgmp
TARGET=gen_parallel_rand
SOURCE=parallel_file_random.cu

# Check if we have CUDA available
CUDA_CHECK := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_CHECK),)
    $(error CUDA compiler (nvcc) not found. Please install CUDA toolkit)
endif

# Check if we have GMP library
GMP_CHECK := $(shell pkg-config --exists gmp && echo "found" || echo "not found")
ifeq ($(GMP_CHECK),not found)
    $(warning GMP library may not be available. Install with: sudo apt-get install libgmp-dev)
endif

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling parallel random number generator with CUDA and OpenMP..."
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
	@echo "Build successful!"
	@echo ""
	@echo "Usage: ./$(TARGET) [num_files] [num_gpus]"
	@echo "Example: ./$(TARGET) 20 4    # Process 20 files using 4 GPUs"

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
	rm -f random_numbers_*.txt
	@echo "Clean complete!"

# Test with small parameters
test: $(TARGET)
	@echo "Running test with 5 files and 1 GPU..."
	./$(TARGET) 5 1

# Performance test with larger parameters
benchmark: $(TARGET)
	@echo "Running benchmark with 50 files and all available GPUs..."
	./$(TARGET) 50 4

# Show GPU information
gpu-info:
	@echo "CUDA GPU Information:"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not found"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@echo -n "CUDA compiler: "
	@which nvcc >/dev/null 2>&1 && echo "✓ Found" || echo "✗ Not found"
	@echo -n "GMP library: "
	@pkg-config --exists gmp && echo "✓ Found" || echo "✗ Not found (install with: sudo apt-get install libgmp-dev)"
	@echo -n "OpenMP: "
	@echo '#include <omp.h>' | gcc -fopenmp -x c - -o /tmp/omp_test 2>/dev/null && echo "✓ Found" || echo "✗ Not found"
	@rm -f /tmp/omp_test

# Create sample files for testing
create-samples:
	@echo "Creating sample input files..."
	@for i in $$(seq 0 9); do \
		echo "$$(($$i * 1000 + 12345))" > random_numbers_$$i.txt; \
	done
	@echo "Created 10 sample files: random_numbers_0.txt to random_numbers_9.txt"

# View results
show-results:
	@echo "Current random number files:"
	@ls -la random_numbers_*.txt 2>/dev/null || echo "No result files found"
	@echo ""
	@echo "Sample content (first 3 files):"
	@for i in 0 1 2; do \
		if [ -f random_numbers_$$i.txt ]; then \
			echo "=== File random_numbers_$$i.txt ==="; \
			head -c 100 random_numbers_$$i.txt; \
			echo "..."; \
		fi; \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build the parallel random number generator"
	@echo "  clean         - Remove built files and generated random files"
	@echo "  test          - Run a small test (5 files, 1 GPU)"
	@echo "  benchmark     - Run performance test (50 files, 4 GPUs)"
	@echo "  gpu-info      - Show available GPU information"
	@echo "  check-deps    - Check if all dependencies are installed"
	@echo "  create-samples- Create sample input files for testing"
	@echo "  show-results  - Display generated random number files"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage: ./$(TARGET) [num_files] [num_gpus]"
	@echo "  num_files     - Number of files to process in parallel (default: 10)"
	@echo "  num_gpus      - Number of GPUs to use (default: 1)"

.PHONY: all clean test benchmark gpu-info check-deps create-samples show-results help