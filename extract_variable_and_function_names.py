import re

def extract_c_identifiers(filename):
    variables = []
    function_names = []
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Remove comments to avoid false positives
    content = re.sub(r'//.*', '', content)  # Remove single-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Remove multi-line comments
    
    # Pattern for variable declarations (including pointers and arrays)
    variable_patterns = [
        r'\b(?:int|float|double|char|void|uint32_t|uint64_t|size_t|mpz_t|cudaError_t|GPUInfo|MontgomeryParams|GPUWorkload|ResidueGroup|cudaStream_t|cudaDeviceProp|FILE)\s+(\*?\s*)(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|;)',
        r'\bstd::\w+\s+(\w+)\s*;'
    ]
    
    # Pattern for function declarations
    function_pattern = r'^(?!__device__|__global__)(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*(?:\{|$)'
    device_function_pattern = r'^(?:__device__|__global__)\s+(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract variables
        for pattern in variable_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if isinstance(match, tuple):
                    if match[1]:  # variable name is in second group
                        variables.append(match[1])
                else:
                    variables.append(match)
        
        # Extract regular functions
        func_match = re.search(function_pattern, line)
        if func_match:
            function_names.append(func_match.group(1))
            
        # Extract device functions
        device_func_match = re.search(device_function_pattern, line)
        if device_func_match:
            function_names.append(device_func_match.group(1))
    
    # Remove duplicates and sort
    variables = sorted(list(set(variables)))
    function_names = sorted(list(set(function_names)))
    
    return variables, function_names

# Use the function
filename = "cuda.c"  # Replace with your actual filename
variables, function_names = extract_c_identifiers(filename)

print("VARIABLES:")
for var in variables:
    print(f"  {var}")

print("\nFUNCTION NAMES:")
for func in function_names:
    print(f"  {func}")

"""

"""