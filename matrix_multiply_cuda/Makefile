CC=nvcc
CFLAGS=-O3 -Xcompiler -fopenmp
#CFLAGS=-O3 -arch=sm_50	# 35, 50, 52, 60
LIBS=-lgmp
TARGET=matrix_multiply_configurable
SOURCE=matrix_multiply_configurable.cu
HEADER=matrix_config.h

# Check if we have CUDA available
CUDA_CHECK := $(shell which nvcc 2>/dev/null)
ifeq ($(CUDA_CHECK),)
    $(error CUDA compiler (nvcc) not found. Please install CUDA toolkit)
endif

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADER)
	@echo "Compiling configurable parallel matrix multiplication with CUDA..."
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)
	@echo "Build successful!"
	@echo ""
	@echo "Usage Examples:"
	@echo "  ./$(TARGET)                              # Default config"
	@echo "  ./$(TARGET) 8 2 512 1024 256            # Custom: 8 tasks, 2 GPUs, dimensions"
	@echo "  ./$(TARGET) --preset small               # Small matrices preset"
	@echo "  ./$(TARGET) --preset large               # Large matrices preset"
	@echo "  ./$(TARGET) --config my_config.txt       # Load from file"
	@echo "  ./$(TARGET) --random 64 1024 16 4        # Random dims 64-1024, 16 tasks, 4 GPUs"

clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
	rm -f matrix_*.txt
	rm -f small_matrix_*.txt
	rm -f medium_matrix_*.txt
	rm -f large_matrix_*.txt
	rm -f rect_matrix_*.txt
	rm -f stress_matrix_*.txt
	rm -f current_config.txt
	rm -f *.config
	@echo "Clean complete!"

# Test with small matrices
test: $(TARGET)
	@echo "Running test with 2 tasks, 1 GPU, 64x64 matrices..."
	./$(TARGET) 2 1 64 64 64

# Medium test
test-medium: $(TARGET)
	@echo "Running medium test with 4 tasks, 2 GPUs, 256x256 matrices..."
	./$(TARGET) 4 2 256 256 256

# Performance benchmark
benchmark: $(TARGET)
	@echo "Running benchmark with 8 tasks, 4 GPUs, 1024x1024 matrices..."
	./$(TARGET) 8 4 1024 1024 1024

# Large scale test
test-large: $(TARGET)
	@echo "Running large test with 16 tasks, 4 GPUs, 512x512 matrices..."
	./$(TARGET) 16 4 512 512 512

# Preset tests
test-presets: $(TARGET)
	@echo "Testing all presets..."
	@echo "Small preset:"
	./$(TARGET) --preset small
	@echo "Medium preset:"
	./$(TARGET) --preset medium
	@echo "Large preset:"
	./$(TARGET) --preset large

# Random configuration test
test-random: $(TARGET)
	@echo "Testing random configuration (256-1024 dimensions, 8 tasks, 2 GPUs):"
	./$(TARGET) --random 256 1024 8 2

# Show GPU information
gpu-info:
	@echo "CUDA GPU Information:"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not found"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@echo -n "CUDA compiler: "
	@which nvcc >/dev/null 2>&1 && echo "✓ Found" || echo "✗ Not found"
	@echo -n "OpenMP: "
	@echo '#include <omp.h>' | gcc -fopenmp -x c - -o /tmp/omp_test 2>/dev/null && echo "✓ Found" || echo "✗ Not found"
	@rm -f /tmp/omp_test

# Create sample matrices for testing
create-samples:
	@echo "Creating sample input matrices..."
	@echo "3 3" > matrix_M_0.txt
	@echo "1.0 2.0 3.0" >> matrix_M_0.txt
	@echo "4.0 5.0 6.0" >> matrix_M_0.txt
	@echo "7.0 8.0 9.0" >> matrix_M_0.txt
	@echo "3 3" > matrix_N_0.txt
	@echo "9.0 8.0 7.0" >> matrix_N_0.txt
	@echo "6.0 5.0 4.0" >> matrix_N_0.txt
	@echo "3.0 2.0 1.0" >> matrix_N_0.txt
	@echo "Created sample 3x3 matrices: matrix_M_0.txt and matrix_N_0.txt"

# View results
show-results:
	@echo "Current matrix files:"
	@ls -la matrix_*.txt small_matrix_*.txt medium_matrix_*.txt large_matrix_*.txt rect_matrix_*.txt stress_matrix_*.txt 2>/dev/null || echo "No matrix files found"
	@echo ""
	@echo "Sample results (first few files):"
	@for prefix in matrix small_matrix medium_matrix large_matrix; do \
		if [ -f $${prefix}_P_0.txt ]; then \
			echo "=== Result Matrix $${prefix}_P_0 ==="; \
			head -5 $${prefix}_P_0.txt; \
			echo "..."; \
		fi; \
	done

# Validate a specific matrix multiplication result
validate-result:
	@if [ -f matrix_M_0.txt ] && [ -f matrix_N_0.txt ] && [ -f matrix_P_0.txt ]; then \
		echo "Validating matrix multiplication result for task 0..."; \
		python3 -c "import numpy as np; \
		M = np.loadtxt('matrix_M_0.txt', skiprows=1); \
		N = np.loadtxt('matrix_N_0.txt', skiprows=1); \
		P_expected = M @ N; \
		P_actual = np.loadtxt('matrix_P_0.txt', skiprows=1); \
		diff = np.abs(P_expected - P_actual).max(); \
		print(f'Maximum difference: {diff}'); \
		print('VALIDATION PASSED' if diff < 1e-5 else 'VALIDATION FAILED')"; \
	else \
		echo "Matrix files not found. Run a test first."; \
	fi

# Performance analysis
perf-analysis: $(TARGET)
	@echo "Running performance analysis..."
	@echo "Small matrices (64x64):"
	@time ./$(TARGET) 4 1 64 64 64
	@echo ""
	@echo "Medium matrices (256x256):"
	@time ./$(TARGET) 4 1 256 256 256
	@echo ""
	@echo "Large matrices (512x512):"
	@time ./$(TARGET) 4 1 512 512 512

# Performance comparison using presets
perf-comparison: $(TARGET)
	@echo "Performance comparison across presets:"
	@echo "=== Small preset ==="
	@time ./$(TARGET) --preset small
	@echo ""
	@echo "=== Medium preset ==="
	@time ./$(TARGET) --preset medium
	@echo ""
	@echo "=== Large preset ==="
	@time ./$(TARGET) --preset large

# Configuration management
config-examples: $(TARGET)
	@echo "Creating example configurations..."
	./$(TARGET) --preset small
	@cp current_config.txt small_example.config
	./$(TARGET) --preset medium
	@cp current_config.txt medium_example.config
	./$(TARGET) --preset large
	@cp current_config.txt large_example.config
	@echo "Example configurations saved as *.config files"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the configurable matrix multiplication program"
	@echo "  clean            - Remove built files and generated matrix files"
	@echo "  test             - Run small test (2 tasks, 1 GPU, 64x64 matrices)"
	@echo "  test-medium      - Run medium test (4 tasks, 2 GPUs, 256x256 matrices)"
	@echo "  test-large       - Run large test (16 tasks, 4 GPUs, 512x512 matrices)"
	@echo "  test-presets     - Test all predefined configurations"
	@echo "  test-random      - Test random configuration generation"
	@echo "  benchmark        - Run performance benchmark (8 tasks, 4 GPUs, 1024x1024)"
	@echo "  gpu-info         - Show available GPU information"
	@echo "  check-deps       - Check if all dependencies are installed"
	@echo "  create-samples   - Create sample input matrices for testing"
	@echo "  show-results     - Display generated matrix files"
	@echo "  validate-result  - Validate matrix multiplication accuracy (requires Python/NumPy)"
	@echo "  perf-analysis    - Run performance analysis with different matrix sizes"
	@echo "  perf-comparison  - Compare performance across presets"
	@echo "  config-examples  - Generate example configuration files"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  ./$(TARGET)                              # Use default configuration"
	@echo "  ./$(TARGET) 8 2 512 1024 256            # Custom: 8 tasks, 2 GPUs, custom dims"
	@echo "  ./$(TARGET) --preset small               # Use small matrices preset"
	@echo "  ./$(TARGET) --preset medium              # Use medium matrices preset"
	@echo "  ./$(TARGET) --preset large               # Use large matrices preset"
	@echo "  ./$(TARGET) --preset rectangular        # Use rectangular matrices preset"
	@echo "  ./$(TARGET) --preset stress              # Use stress testing preset"
	@echo "  ./$(TARGET) --config my_config.txt       # Load configuration from file"
	@echo "  ./$(TARGET) --random 64 1024 16 4        # Random dims 64-1024, 16 tasks, 4 GPUs"

.PHONY: all clean test test-medium test-large test-presets test-random benchmark gpu-info check-deps create-samples show-results validate-result perf-analysis perf-comparison config-examples help
	