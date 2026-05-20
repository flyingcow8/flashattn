#!/bin/bash

# Define color constants for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

declare -A arch_map_ori=(
    ["c500"]="--offload-arch=xcore1000"
    ["c600"]="--offload-arch=xcore1500"
)

declare -A arch_map
for key in "${!arch_map_ori[@]}"; do
    arch_map[${key,,}]="${arch_map_ori[$key]}"
done

# Function to get torch installation path from pip
function get_torch_path() {
    # Query pip for torch package info and extract location path
    pip show torch | grep Location | awk '{print $2}'
}

# Function to create backup of target file
function backup_file() {
    local src=$1  # Source file path
    # Create backup with .bak extension
    cp "$src" "$src.bak" || return 1  # Return error if backup fails
}

# Function to modify MXCC flags in the target file
function modify_flags() {
    local target_file=$1  # File to modify
    local target_flags=$2  # Flags to be set
    if grep -q $'\r' $target_file; then
        sed -i 's/\x0D//g' $target_file
    fi
    raw_flags="$target_flags"
    escaped_flags=$(printf '%s' "$target_flags" | \
    awk '{
        gsub(/\\/, "\\\\")
        gsub(/\047/, "\\\047")
        gsub(/\[/, "\\[")
        gsub(/\]/, "\\]")
        print
    }')

    temp_file=$(mktemp)
    trap 'rm -f "$temp_file"' EXIT

    # sed replacement logic (preserves pattern space)
    sed -e '/^[[:space:]]*COMMON_MXCC_FLAGS[[:space:]]*=[[:space:]]*(/,/^[[:space:]]*)/{
    /^[[:space:]]*COMMON_MXCC_FLAGS[[:space:]]*=/{
        s|.*|COMMON_MXCC_FLAGS = (\n[\x27-DUSE_MACA\x27,\x27-DNV_ARCH_A100\x27,'"$escaped_flags"']\n)|
        b done
    }
    d
    :done
    n
    /^[[:space:]]*)/!d
    }' "$target_file" > "$temp_file"

    validate_content() {
    local validation_passed=true
    # Check required base flag
    if ! grep -q "'-DUSE_MACA'" "$temp_file"; then
        echo -e "${RED}[ERROR] Validation failed: base flag missing${NC}" >&2
        validation_passed=false
    fi
    $validation_passed
    }

    # Execute validation and replacement
    if validate_content; then
        if ! diff -q "$target_file" "$temp_file" >/dev/null; then
            mv "$temp_file" "$target_file"
            echo -e "${GREEN}[SUCCESS] Set PyTorch extension arch flags based on environment variable FLASHATTN_BUILD_PROJECTS=$FLASHATTN_BUILD_PROJECTS${NC}" && exit 0
        else
            echo -e "${GREEN}[INFO] No modification to the PyTorch extension arch flags${NC}"
        fi
    else
        echo -e "${RED}[ERROR] Failed to set PyTorch extension arch flags: validation not passed${NC}" >&2
        exit 1
    fi

}


# Function to display help information
function print_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 <Process_type> [options]"
    echo -e "${YELLOW}Supported process types:${NC}"
    echo -e "  set\tSet PyTorch extension arch flags based on environment variable FLASHATTN_BUILD_PROJECTS"
    echo -e "  get\tGet PyTorch extension arch flags to set environment variable FLASHATTN_BUILD_PROJECTS"
    echo -e "  restore\tRestore PyTorch extension script"
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  -h, --help\tShow this help message"
}

function convert_archs_to_flags() {
    local projects=${FLASHATTN_BUILD_PROJECTS,,}
    IFS=':' read -ra archs <<< "$projects"
    flags=()
    for arch in "${archs[@]}"; do
        if [[ -n "${arch_map[${arch,,}]}" ]]; then
            flags+=("'${arch_map[${arch,,}]}'")
        fi
    done
    if [[ ${#flags[@]} -gt 0 ]]; then
        flag_str=$(IFS=','; printf "%s" "${flags[*]}")
        echo "$flag_str"
    else
        echo -e "${RED}[ERROR] No matching conversion flags found${NC}" >&2
        echo ""
        return
    fi
}

function convert_flags_to_archs() {
    local target_file=$1  # File to read
    flags_content=$(grep -A10 "COMMON_MXCC_FLAGS = (" "$target_file" | grep -A10 "\[" | grep -B10 "\]" | grep -v "\[" | grep -v "\]")

    # Process each line containing --offload-arch= parameter
    processed_flags=""
    while read -r line; do
        if [[ $line == *"--offload-arch="* ]]; then
            arch_value=${line#*--offload-arch=}
            # Convert architecture values according to requirements
            case $arch_value in
                *xcore1000*) converted="C500" ;;
                *xcore1500*) converted="C600" ;;
                *) converted=$arch_value ;;
            esac
            if [ -z "$processed_flags" ]; then
                processed_flags="$converted"
            else
                processed_flags="$processed_flags:$converted"
            fi
        fi
    done <<< "$flags_content"

    echo "$processed_flags"
}

# Function to set torch extension arch flags
function set_extension_flags() {
    echo -e "${GREEN}[INFO] Start setting PyTorch extension arch flags...${NC}"
    # Get torch installation path
    torch_path=$(get_torch_path)
    [ -z "$torch_path" ] && { echo -e "${RED}[ERROR] Torch not found${NC}"; exit 1; }
    # Define target file cpp_extension.py path
    target_file="$torch_path/torch/utils/cpp_extension.py"
    [ -f "$target_file" ] || { echo -e "${RED}[ERROR] Target file not found${NC}"; exit 1; }
    # Create backup before modification
    backup_file "$target_file" || { echo -e "${RED}[ERROR] Backup failed${NC}"; exit 1; }
    # Convert archs to flags
    flags=$(convert_archs_to_flags)
    if [[ -n "$flags" ]]; then
        # Perform the flag modification
        modify_flags "$target_file" "$flags" || {  echo -e "${RED}[ERROR] Modification failed${NC}"; exit 1; }
    fi
}

# Function to get torch extension arch flags
function get_extension_flags() {
    echo -e "${GREEN}[INFO] Start getting PyTorch extension arch flags...${NC}"
    # Get torch installation path
    torch_path=$(get_torch_path)
    [ -z "$torch_path" ] && { echo -e "${RED}[ERROR] Torch not found${NC}"; exit 1; }
    # Define target file cpp_extension.py path
    target_file="$torch_path/torch/utils/cpp_extension.py"
    [ -f "$target_file" ] || { echo -e "${RED}[ERROR] Target file not found${NC}"; exit 1; }
    archs=$(convert_flags_to_archs $target_file)
    # Set default if empty
    if [[ -z "$archs" ]]; then
       archs="C500:C600"
    fi
    echo -e "${GREEN}[INFO] Get PyTorch extension arch flags completed${NC}" >&2
    echo $archs
}

main() {
    # Check if any argument was provided
    if [ $# -eq 0 ]; then
        echo -e "${RED}[ERROR] No process type specified${NC}"
        print_help
        exit 1
    fi

    case "$1" in
        set)
            set_extension_flags
            ;;
        get)
            get_extension_flags
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown process type: $1${NC}"
            print_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
