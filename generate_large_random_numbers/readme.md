
```
% ssh_run_cuda "generate_large_random_numbers" 2 5 "cuda_results.log"
[2025-09-05 19:42:52] Starting automated run for folder: generate_large_random_numbers
[2025-09-05 19:42:52] Zipping and sending folder...
[2025-09-05 19:42:52] Folder sent successfully
[2025-09-05 19:42:52] Unzipping on remote...
Archive:  generate_large_random_numbers.zip
   creating: generate_large_random_numbers/
  inflating: generate_large_random_numbers/Makefile  
  inflating: generate_large_random_numbers/generate_large_random_numbers.cu  
[2025-09-05 19:42:52] Unzipped successfully
[2025-09-05 19:42:52] Running command: make clean && make && ./gen_large_rand 2 5
rm -f gen_large_rand
nvcc -O3 generate_large_random_numbers.cu -o gen_large_rand -lgmp
Using 2 GPU(s) to generate 5 random numbers of 4000 bits each

Generated 4000-bit random numbers:
1: 7078255452371516746...
:
5: 9188348732740619465 ... 421
[2025-09-05 19:42:52] Retrieving logs...
[2025-09-05 19:42:52] Logs retrieved and appended to cuda_results.log
[2025-09-05 19:42:52] Automated run completed
```

```
% ssh_run_and_collect "generate_and_add_large_random_numbers" \
    "make gpu-info && make check-deps" \
    "cuda_system_info.log"
[2025-09-05 21:45:56] Starting automated run for folder: generate_and_add_large_random_numbers
[2025-09-05 21:45:56] Zipping and sending folder...
[2025-09-05 21:45:56] Folder sent successfully
[2025-09-05 21:45:56] Unzipping on remote...
Archive:  generate_and_add_large_random_numbers.zip
   creating: generate_and_add_large_random_numbers/
  inflating: generate_and_add_large_random_numbers/Makefile  
  inflating: generate_and_add_large_random_numbers/readme.md  
  inflating: generate_and_add_large_random_numbers/generate_and_add_large_random_numbers.cu  
[2025-09-05 21:45:56] Unzipped successfully
[2025-09-05 21:45:56] Running command: make gpu-info && make check-deps
CUDA GPU Information:
0, Quadro M4000, 8192, 8108
1, Quadro M4000, 8192, 8108
Checking dependencies...
CUDA compiler: ✓ Found
GMP library: ✓ Found
OpenMP: ✗ Not found
[2025-09-05 21:45:56] Retrieving logs...
[2025-09-05 21:45:56] Logs retrieved and appended to cuda_system_info.log
[2025-09-05 21:45:56] Automated run completed
```