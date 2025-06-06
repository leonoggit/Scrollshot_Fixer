#!/bin/bash

# Real-time VS Code Error Monitor
# This script monitors VS Code Insiders for errors and provides real-time feedback

echo "🔍 Starting VS Code Error Monitor..."
echo "📅 $(date)"
echo "======================================"

# Function to check VS Code processes
check_vscode_processes() {
    echo "📊 VS Code Processes:"
    ps aux | grep -E "(code-insiders|code)" | grep -v grep || echo "   No VS Code processes found"
    echo ""
}

# Function to monitor extension host crashes
monitor_extension_host() {
    echo "🔧 Checking Extension Host Status..."
    
    # Check if extension host is running
    if pgrep -f "extensionHost" > /dev/null; then
        echo "   ✅ Extension Host is running"
    else
        echo "   ⚠️  Extension Host may have crashed or not started"
    fi
    echo ""
}

# Function to check VS Code logs for common errors
check_vscode_logs() {
    echo "📋 Recent VS Code Errors (last 10 lines):"
    
    # Common VS Code log locations
    LOG_DIRS=(
        "$HOME/.config/Code - Insiders/logs"
        "$HOME/.vscode-insiders/logs"
        "/tmp/vscode-logs"
    )
    
    found_logs=false
    for log_dir in "${LOG_DIRS[@]}"; do
        if [ -d "$log_dir" ]; then
            echo "   📂 Checking: $log_dir"
            find "$log_dir" -name "*.log" -type f -mtime -1 2>/dev/null | head -3 | while read logfile; do
                if [ -f "$logfile" ]; then
                    echo "   📄 Recent errors in $(basename "$logfile"):"
                    grep -i -E "(error|exception|failed|crash)" "$logfile" 2>/dev/null | tail -5 | sed 's/^/      /'
                    found_logs=true
                fi
            done
        fi
    done
    
    if [ "$found_logs" = false ]; then
        echo "   ℹ️  No recent log files found"
    fi
    echo ""
}

# Function to check workspace-specific issues
check_workspace_issues() {
    echo "🏗️  Workspace Configuration Check:"
    
    WORKSPACE_DIR="/workspaces/Scrollshot_Fixer"
    
    if [ -f "$WORKSPACE_DIR/.vscode/settings.json" ]; then
        echo "   📋 Checking settings.json for problematic configurations..."
        
        # Check for empty file path references
        if grep -q '${file}' "$WORKSPACE_DIR/.vscode/settings.json" 2>/dev/null; then
            echo "   ⚠️  Found '${file}' variable references - potential source of empty file errors"
        fi
        
        # Check for problematic formatter settings
        if grep -q '".*format.*".*:.*""' "$WORKSPACE_DIR/.vscode/settings.json" 2>/dev/null; then
            echo "   ⚠️  Found empty formatter settings"
        fi
        
        echo "   ✅ Settings.json syntax check:"
        if python3 -m json.tool "$WORKSPACE_DIR/.vscode/settings.json" > /dev/null 2>&1; then
            echo "      Valid JSON syntax"
        else
            echo "      ❌ Invalid JSON syntax detected!"
        fi
    else
        echo "   ℹ️  No .vscode/settings.json found"
    fi
    echo ""
}

# Function to check extension status
check_extension_status() {
    echo "🧩 Extension Status Check:"
    
    # Try to get extension list (this may not work in Codespaces)
    if command -v code-insiders >/dev/null 2>&1; then
        echo "   📦 Attempting to list extensions..."
        timeout 10s code-insiders --list-extensions 2>/dev/null | head -10 | sed 's/^/      /' || echo "      ⚠️  Could not retrieve extension list"
    else
        echo "   ⚠️  code-insiders command not available"
    fi
    echo ""
}

# Function to monitor file system events that might trigger errors
monitor_file_events() {
    echo "📁 Recent File System Activity:"
    
    WORKSPACE_DIR="/workspaces/Scrollshot_Fixer"
    
    # Check for recent file modifications that might trigger extension errors
    echo "   📝 Recently modified files (last 5 minutes):"
    find "$WORKSPACE_DIR" -type f -mmin -5 2>/dev/null | grep -E '\.(swift|m|mm|h|json|yml|yaml)$' | head -5 | sed 's/^/      /' || echo "      No recent modifications"
    echo ""
}

# Function to provide error resolution suggestions
provide_suggestions() {
    echo "💡 Error Resolution Suggestions:"
    echo "   1. If you see 'file argument empty' errors:"
    echo "      - Check .vscode/settings.json for '${file}' variables"
    echo "      - Disable problematic formatters"
    echo "      - Restart VS Code Insiders"
    echo ""
    echo "   2. If Extension Host crashes:"
    echo "      - Disable recently installed extensions"
    echo "      - Clear extension cache"
    echo "      - Check for conflicting extensions"
    echo ""
    echo "   3. For iOS development issues:"
    echo "      - Ensure Swift extensions are compatible"
    echo "      - Check Xcode command line tools"
    echo "      - Verify iOS simulator access"
    echo ""
}

# Main monitoring loop
main_monitor() {
    while true; do
        clear
        echo "🔍 VS Code Real-time Error Monitor"
        echo "📅 $(date)"
        echo "🔄 Press Ctrl+C to stop monitoring"
        echo "======================================"
        echo ""
        
        check_vscode_processes
        monitor_extension_host
        check_vscode_logs
        check_workspace_issues
        check_extension_status
        monitor_file_events
        provide_suggestions
        
        echo "🔄 Next check in 30 seconds..."
        sleep 30
    done
}

# Check if running in continuous mode
if [ "$1" = "--continuous" ] || [ "$1" = "-c" ]; then
    main_monitor
else
    # Single run mode
    check_vscode_processes
    monitor_extension_host
    check_vscode_logs
    check_workspace_issues
    check_extension_status
    monitor_file_events
    provide_suggestions
    
    echo "💻 To run continuous monitoring: ./error_monitor.sh --continuous"
fi
