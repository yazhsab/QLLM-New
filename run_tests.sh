#!/bin/bash

# Run all tests for the Qynthra project

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "\n${YELLOW}=================================================="
    echo -e "$1"
    echo -e "==================================================${NC}"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Run a test and check the result
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    print_header "Running $test_name"
    
    # Run the test command
    eval "$test_command"
    local result=$?
    
    # Check the result
    if [ $result -eq 0 ]; then
        print_success "$test_name completed successfully"
        return 0
    else
        print_error "$test_name failed with exit code $result"
        return 1
    fi
}

# Main function
main() {
    print_header "Qynthra Test Suite"
    
    # Check if Python is installed
    if ! command_exists python; then
        print_error "Python is not installed"
        exit 1
    fi
    
    # Check if pip is installed
    if ! command_exists pip; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Step 1: Run structure test
    run_test "Structure Test" "python test_qynthra.py"
    
    # Step 2: Install the package in development mode
    print_header "Installing Qynthra package"
    pip install -e .
    
    # Step 3: Run functional tests
    run_test "Functional Tests" "python functional_test.py"
    
    # Step 4: Run example tests
    run_test "Example Tests" "python test_examples.py"
    
    # Step 5: Open the JavaScript client test in a browser
    print_header "Testing JavaScript Client"
    
    # Check if we can open a browser
    if command_exists open; then
        # macOS
        open test_js_client.html
        print_success "Opened JavaScript client test in browser"
    elif command_exists xdg-open; then
        # Linux
        xdg-open test_js_client.html
        print_success "Opened JavaScript client test in browser"
    elif command_exists start; then
        # Windows
        start test_js_client.html
        print_success "Opened JavaScript client test in browser"
    else
        print_error "Could not open browser. Please open test_js_client.html manually."
    fi
    
    print_header "Test Suite Completed"
    echo "To run individual tests:"
    echo "  python test_qynthra.py      # Test project structure"
    echo "  python functional_test.py   # Test functionality"
    echo "  python test_examples.py     # Test examples"
    echo "  open test_js_client.html    # Test JavaScript client"
}

# Run the main function
main