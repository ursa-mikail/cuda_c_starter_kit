
```
Complete Automation: Zips → Sends → Unzips → Runs → Retrieves logs
Timestamped Logs: Each run is clearly marked with timestamps
Append-Only Logging: Preserves all previous runs in the log file
Error Handling: Checks for failures at each step
CUDA Convenience Function: ssh_run_cuda specifically for GPU projects
Flexible Commands: Can run any command, not just CUDA programs

The logs will contain the complete output from your remote runs, making it easy to track performance across different GPU configurations and parameters!
```

functions at: [shell_script_utility](https://github.com/ursa-mikail/shell_script_utility)
in [dev_shell.sh](https://github.com/ursa-mikail/shell_script_utility/blob/main/scripts/utilities/dev_shell.sh)
```
# Zip local folder and send it to remote
function ssh_zip_folder_and_send() 

# Extended function: zip, send, run, and get logs back
function ssh_run_and_collect() 

# Convenience function specifically for CUDA projects
function ssh_run_cuda() 

# Function to just retrieve and append existing remote logs
function ssh_get_logs() 
```