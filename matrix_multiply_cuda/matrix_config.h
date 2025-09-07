#ifndef MATRIX_CONFIG_H
#define MATRIX_CONFIG_H

// Matrix configuration structure
typedef struct {
    int M_rows;       // Matrix M rows
    int M_cols;       // Matrix M columns (must equal N_rows for multiplication)
    int N_rows;       // Matrix N rows (must equal M_cols)
    int N_cols;       // Matrix N columns
    int P_rows;       // Result matrix P rows (equals M_rows)
    int P_cols;       // Result matrix P columns (equals N_cols)
    
    // Random generation parameters
    float min_value;  // Minimum random value
    float max_value;  // Maximum random value
    
    // CUDA parameters
    int block_size;   // CUDA block size (e.g., 16 for 16x16)
    
    // Task parameters
    int num_tasks;    // Number of parallel tasks
    int num_gpus;     // Number of GPUs to use
    
    // File naming
    char file_prefix[64];  // Prefix for matrix files
} MatrixConfig;

// Default configuration
static const MatrixConfig DEFAULT_CONFIG = {
    .M_rows = 256,
    .M_cols = 512,
    .N_rows = 512,     // Must equal M_cols
    .N_cols = 128,
    .P_rows = 256,     // Equals M_rows
    .P_cols = 128,     // Equals N_cols
    
    .min_value = 0.0f,
    .max_value = 100.0f,
    
    .block_size = 16,
    
    .num_tasks = 4,
    .num_gpus = 1,
    
    .file_prefix = "matrix"
};

// Predefined configurations for different scenarios

// Small matrices for testing
static const MatrixConfig SMALL_CONFIG = {
    .M_rows = 64,
    .M_cols = 32,
    .N_rows = 32,
    .N_cols = 96,
    .P_rows = 64,
    .P_cols = 96,
    
    .min_value = -10.0f,
    .max_value = 10.0f,
    
    .block_size = 8,
    
    .num_tasks = 2,
    .num_gpus = 1,
    
    .file_prefix = "small_matrix"
};

// Medium matrices for benchmarking
static const MatrixConfig MEDIUM_CONFIG = {
    .M_rows = 512,
    .M_cols = 256,
    .N_rows = 256,
    .N_cols = 1024,
    .P_rows = 512,
    .P_cols = 1024,
    
    .min_value = 0.0f,
    .max_value = 50.0f,
    
    .block_size = 16,
    
    .num_tasks = 8,
    .num_gpus = 2,
    
    .file_prefix = "medium_matrix"
};

// Large matrices for performance testing
static const MatrixConfig LARGE_CONFIG = {
    .M_rows = 1024,
    .M_cols = 2048,
    .N_rows = 2048,
    .N_cols = 512,
    .P_rows = 1024,
    .P_cols = 512,
    
    .min_value = -100.0f,
    .max_value = 100.0f,
    
    .block_size = 32,
    
    .num_tasks = 16,
    .num_gpus = 4,
    
    .file_prefix = "large_matrix"
};

// Rectangular matrices for specialized testing
static const MatrixConfig RECTANGULAR_CONFIG = {
    .M_rows = 128,
    .M_cols = 2048,
    .N_rows = 2048,
    .N_cols = 64,
    .P_rows = 128,
    .P_cols = 64,
    
    .min_value = -1.0f,
    .max_value = 1.0f,
    
    .block_size = 16,
    
    .num_tasks = 6,
    .num_gpus = 3,
    
    .file_prefix = "rect_matrix"
};

// Very large matrices for stress testing
static const MatrixConfig STRESS_CONFIG = {
    .M_rows = 2048,
    .M_cols = 4096,
    .N_rows = 4096,
    .N_cols = 1024,
    .P_rows = 2048,
    .P_cols = 1024,
    
    .min_value = -1000.0f,
    .max_value = 1000.0f,
    
    .block_size = 32,
    
    .num_tasks = 32,
    .num_gpus = 8,
    
    .file_prefix = "stress_matrix"
};

// Function prototypes for configuration management
MatrixConfig load_config_from_file(const char* filename);
void save_config_to_file(const MatrixConfig* config, const char* filename);
MatrixConfig parse_config_from_args(int argc, char* argv[]);
void print_config(const MatrixConfig* config);
int validate_config(const MatrixConfig* config);
MatrixConfig randomize_config_dimensions(int min_dim, int max_dim, int num_tasks, int num_gpus);

#endif // MATRIX_CONFIG_H
