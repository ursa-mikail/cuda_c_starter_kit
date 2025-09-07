# Parallel Large Random Number Generator with File I/O

This project generates large 4000-bit random numbers in parallel across multiple GPUs, reads previous values from files, adds them to newly generated numbers, and writes the results back to individual files.

## Features

- **Multi-GPU Support**: Distributes work across multiple CUDA-capable GPUs
- **Multi-Process Parallelism**: Uses fork() for true parallel file processing
- **Large Number Arithmetic**: Handles 4000-bit numbers using GMP library
- **Persistent State**: Reads previous values from files and adds to new random numbers
- **Scalable**: Process N files with M GPUs efficiently

## Prerequisites

- CUDA-capable GPU(s)
- CUDA Toolkit (nvcc compiler)
- GMP library for large number arithmetic
- Linux/Unix system with fork() support

### Installation of Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit libgmp-dev build-essential

# CentOS/RHEL
sudo yum install cuda-toolkit gmp-devel gcc-c++

# Check if everything is installed
make check-deps
```

## Building

```bash
# Build the project
make

# Check dependencies first
make check-deps

# Clean and rebuild
make clean && make
```

## Usage

### Basic Usage

```bash
# Process 10 files using 1 GPU (default)
./gen_parallel_rand

# Process 20 files using 2 GPUs
./gen_parallel_rand 20 2

# Process 100 files using 4 GPUs
./gen_parallel_rand 100 4
```

### Parameters

- **num_files**: Number of files to process in parallel (default: 10)
- **num_gpus**: Number of GPUs to utilize (default: 1)

### File Format

- Input/Output files: `random_numbers_0.txt`, `random_numbers_1.txt`, etc.
- Each file contains one large decimal number
- If a file doesn't exist, it starts with 0
- New random number is added to the previous value

## Testing

```bash
# Quick test with 5 files
make test

# Performance benchmark
make benchmark

# Create sample files for testing
make create-samples

# View generated results
make show-results
```

## Remote Execution with SSH

If you have the SSH automation functions set up, you can run this remotely:

```bash
# Basic remote execution
ssh_run_cuda "generate_and_add_large_random_numbers" 2 5 "cuda_generate_and_add_large_random_numbers_results.log"

# For this specific parallel version, use custom command:
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make clean && make && ./gen_parallel_rand 50 4" \
    "cuda_generate_and_add_large_random_numbers_results.log"

# Quick remote test
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make test" \
    "cuda_test_results.log"
```

```
# Basic usage - 20 files, 2 GPUs
ssh_run_cuda "generate_and_add_large_random_numbers" 20 2 "cuda_generate_and_add_large_random_numbers_results.log"

# Custom command with specific parameters
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make clean && make && ./gen_parallel_rand 50 4" \
    "cuda_generate_and_add_large_random_numbers_results.log"

# Quick test run
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make test" \
    "cuda_test_results.log"

# Performance benchmark
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make benchmark" \
    "cuda_benchmark_results.log"

# Check GPU info on remote system
ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make gpu-info && make check-deps" \
    "cuda_system_info.log"
```

## Performance Characteristics

### Scaling with GPUs
- **1 GPU**: Processes files sequentially on single GPU
- **Multiple GPUs**: Distributes processes across GPUs (round-robin)
- **Optimal**: Number of processes ≈ Number of GPU cores / 32

### Scaling with Files
- **Low file count** (< 10): Limited by GPU utilization
- **Medium file count** (10-100): Good balance of parallelism
- **High file count** (> 100): May be limited by CPU context switching

### Typical Performance
```
Files   GPUs   Time (approx)
10      1      5-10 seconds
50      2      15-25 seconds
100     4      30-45 seconds
```

## Example Output

```
$ ./gen_parallel_rand 5 2
Processing 5 files using 2 GPUs in parallel
Process 12345: Using GPU 0 for file random_numbers_0.txt
Process 12346: Using GPU 1 for file random_numbers_1.txt
Process 12347: Using GPU 0 for file random_numbers_2.txt
Process 12348: Using GPU 1 for file random_numbers_3.txt
Process 12349: Using GPU 0 for file random_numbers_4.txt
File 0 processing completed
File 1 processing completed
File 2 processing completed
File 3 processing completed
File 4 processing completed
All files processed successfully!
```

```
% create_and_goto_folder 'generate_and_add_large_random_numbers'
Creating : 'generate_and_add_large_random_numbers'
generate_and_add_large_random_numbers % subl Makefile
generate_and_add_large_random_numbers % subl readme.md
generate_and_add_large_random_numbers % subl generate_and_add_large_random_numbers.cu
generate_and_add_large_random_numbers % cd ..
m % ssh_run_cuda "generate_and_add_large_random_numbers" 20 2 "cuda_generate_and_add_large_random_numbers_results.log"
[2025-09-05 21:46:37] Starting automated run for folder: generate_and_add_large_random_numbers
[2025-09-05 21:46:37] Zipping and sending folder...
[2025-09-05 21:46:37] Folder sent successfully
[2025-09-05 21:46:37] Unzipping on remote...
Archive:  generate_and_add_large_random_numbers.zip
  inflating: generate_and_add_large_random_numbers/Makefile  
  inflating: generate_and_add_large_random_numbers/readme.md  
  inflating: generate_and_add_large_random_numbers/generate_and_add_large_random_numbers.cu  
[2025-09-05 21:46:37] Unzipped successfully
[2025-09-05 21:46:37] Running command: make clean && make && ./gen_large_rand 20 2
Cleaning up...
rm -f gen_parallel_rand
rm -f random_numbers_*.txt
Clean complete!
make: *** No rule to make target 'parallel_file_random.cu', needed by 'gen_parallel_rand'.  Stop.
[2025-09-05 21:46:37] Retrieving logs...
[2025-09-05 21:46:37] Logs retrieved and appended to cuda_generate_and_add_large_random_numbers_results.log
[2025-09-05 21:46:37] Automated run completed
m % ls *.log
cuda_generate_and_add_large_random_numbers_results.log
cuda_results.log
cuda_system_info.log
```

## File Structure

```
generate_and_add_large_random_numbers/
├── parallel_file_random.cu      # Main CUDA source code
├── Makefile                      # Build configuration
├── README.md                     # This file
├── random_numbers_0.txt         # Generated/updated by program
├── random_numbers_1.txt         # Generated/updated by program
└── ...                          # More numbered files
```

## Troubleshooting

### CUDA Errors
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Verify GPU info
make gpu-info
```

### Compilation Errors
```bash
# Check all dependencies
make check-deps

# Clean and rebuild
make clean && make
```

### Runtime Issues
```bash
# Start with smaller parameters
./gen_parallel_rand 2 1

# Check file permissions
ls -la random_numbers_*.txt

# Monitor system resources
top -p $(pgrep gen_parallel_rand)
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make` | Build the project |
| `make clean` | Remove built files and generated random files |
| `make test` | Run small test (5 files, 1 GPU) |
| `make benchmark` | Performance test (50 files, 4 GPUs) |
| `make check-deps` | Verify all dependencies are installed |
| `make gpu-info` | Display available GPU information |
| `make create-samples` | Generate sample input files |
| `make show-results` | Display generated random number files |
| `make help` | Show all available targets |

## Algorithm Details

1. **Fork Process per File**: Each file gets its own process for true parallelism
2. **GPU Assignment**: Processes are assigned to GPUs using round-robin (process_id % num_gpus)
3. **Random Generation**: Each process generates one 4000-bit random number using CUDA
4. **File Operations**: 
   - Read previous number from file (or start with 0)
   - Add newly generated random number to previous value
   - Write result back to file
5. **Synchronization**: Parent process waits for all children to complete

## License

This code is provided as-is for educational and research purposes.
