CC      = nvcc
CFLAGS  = -O3 -arch=sm_50    # adjust sm_XX to your GPU architecture
LIBS    = -lcublas

# Sources and targets
SRCS    = parallel_reduction_add_demo.cu parallel_reduction_add_demo_optimized.cu
TARGETS = $(SRCS:.cu=)

all: $(TARGETS)

%: %.cu
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(TARGETS)

# GPU info
gpu-info:
	@echo "CUDA GPU Information:"
	@nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which nvcc > /dev/null && echo "CUDA compiler: ✓ Found" || echo "CUDA compiler: ✗ Not found"
	@which nvidia-smi > /dev/null && echo "nvidia-smi: ✓ Found" || echo "nvidia-smi: ✗ Not found"

# Run tests on both versions
test-small: $(TARGETS)
	@echo "== Shared Memory Version =="
	./parallel_reduction_add_demo 1 256
	@echo "== Optimized Warp-Shuffle Version =="
	./parallel_reduction_add_demo_optimized 1 256

test-medium: $(TARGETS)
	@echo "== Shared Memory Version =="
	./parallel_reduction_add_demo 1 512
	@echo "== Optimized Warp-Shuffle Version =="
	./parallel_reduction_add_demo_optimized 1 512

test-large: $(TARGETS)
	@echo "== Shared Memory Version =="
	./parallel_reduction_add_demo 1 1024
	@echo "== Optimized Warp-Shuffle Version =="
	./parallel_reduction_add_demo_optimized 1 1024

benchmark: $(TARGETS)
	@echo "=== Performance Benchmark ==="
	@echo "Small input:"
	./parallel_reduction_add_demo 1 256
	./parallel_reduction_add_demo_optimized 1 256
	@echo "Medium input:"
	./parallel_reduction_add_demo 1 512
	./parallel_reduction_add_demo_optimized 1 512
	@echo "Large input:"
	./parallel_reduction_add_demo 1 1024
	./parallel_reduction_add_demo_optimized 1 1024

.PHONY: all clean gpu-info check-deps test-small test-medium test-large benchmark

#make clean && make
#make gpu-info       # check GPU info
#make check-deps     # verify CUDA setup
#make test-small     # run 256 elements on both
#make test-medium    # run 512 elements on both
#make test-large     # run 1024 elements on both
#make benchmark      # run all benchmarks
