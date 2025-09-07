CC=nvcc
CFLAGS=-O3 -arch=sm_50	# 35, 50, 52, 60
LIBS=-lcublas

TARGET=matrix_sync_demo
SOURCE=matrix_sync_demo.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

# Check system info
gpu-info:
	@echo "CUDA GPU Information:"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

check-deps:
	@echo "Checking dependencies..."
	@which nvcc > /dev/null && echo "CUDA compiler: ✓ Found" || echo "CUDA compiler: ✗ Not found"
	@pkg-config --exists libcublas && echo "CUBLAS library: ✓ Found" || echo "CUBLAS library: ✗ Not found"
	@which nvidia-smi > /dev/null && echo "nvidia-smi: ✓ Found" || echo "nvidia-smi: ✗ Not found"

# Test runs
test-small:
	./$(TARGET) 1 256

test-medium:
	./$(TARGET) 1 512

test-large:
	./$(TARGET) 1 1024

test-multi-gpu:
	./$(TARGET) 2 512

# Performance benchmark
benchmark: $(TARGET)
	@echo "=== Performance Benchmark ==="
	@echo "Small matrices (256x256):"
	@./$(TARGET) 1 256
	@echo "Medium matrices (512x512):"
	@./$(TARGET) 1 512
	@echo "Large matrices (1024x1024):"
	@./$(TARGET) 1 1024

.PHONY: all clean gpu-info check-deps test-arch test-small test-medium test-large test-multi-gpu benchmark