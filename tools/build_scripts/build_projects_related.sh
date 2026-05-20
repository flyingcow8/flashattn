#!/bin/bash

# Define color constants for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")

# Function to display help information
function print_help() {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 <project_type> [options]"
    echo -e "${YELLOW}Supported project types:${NC}"
    echo -e "  pytorch\tGet PyTorch build projects"
    echo -e "  sdk\t\tGet SDK build projects"
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  -h, --help\tShow this help message"
}

# Function to get FlashAttn private environment variable
function get_private_build_projects() {
    local projects="${FLASHATTN_BUILD_PROJECTS:-}"
    # Return empty if not set
    [ -z "$projects" ] && echo "" && return
    if [[ "$projects" =~ ^([A-Za-z]?[0-9]{3}[A-Za-z]?)(:[A-Za-z]?[0-9]{3}[A-Za-z]?)*$ ]]; then
        echo "$projects"
    else
        unset FLASHATTN_BUILD_PROJECTS
        echo -e "${YELLOW}[WARNING] FLASHATTN_BUILD_PROJECTS format must be [A-Z]XXX[A-Z] or [A-Z]XXX[A-Z]:[A-Z]XXX[A-Z]...${NC}" >&2
        echo ""
        return
    fi
}

# Function to get SDK common environment variable
function get_common_build_projects() {
    local projects="${BUILD_PROJECTS:-}"
    # Return empty if not set
    [ -z "$projects" ] && echo "" && return
    # Validate pattern
    if [[ "$projects" =~ ^([A-Za-z]?[0-9]{3}[A-Za-z]?)(:[A-Za-z]?[0-9]{3}[A-Za-z]?)*$ ]]; then
        echo "$projects"
    else
        echo -e "${YELLOW}[WARNING] BUILD_PROJECTS format must be [A-Z]XXX[A-Z] or [A-Z]XXX[A-Z]:[A-Z]XXX[A-Z]...${NC}" >&2
        echo ""
        return
    fi
}

# Function to get PyTorch build projects
function get_pytorch_build_projects() {
    flashattn_build_projects=$(get_private_build_projects)
    if [ -z "$flashattn_build_projects" ]; then
        export FLASHATTN_BUILD_PROJECTS="C500:C600"
    fi
    projects=${FLASHATTN_BUILD_PROJECTS:-}
    echo -e "${GREEN}[INFO] Set arch flags in torch extension by: $projects ${NC}"
    bash ${SCRIPT_PATH}/torch_extension_related.sh set
}

# Function to get SDK build projects
function get_sdk_build_projects() {
    echo -e "${GREEN}[INFO] Start getting SDK build projects...${NC}"
    flashattn_build_projects=$(get_private_build_projects)
    if [ -z "$flashattn_build_projects" ]; then
        common_build_projects=$(get_common_build_projects)
        if [ -z "$common_build_projects" ]; then
            export FLASHATTN_BUILD_PROJECTS="C500:C600"
        else
            export FLASHATTN_BUILD_PROJECTS=${common_build_projects}
        fi
    fi
    echo "Get FLASHATTN_BUILD_PROJECTS value: $FLASHATTN_BUILD_PROJECTS"
    echo -e "${GREEN}[INFO] Get SDK build projects completed${NC}"
}


main() {
    # Check if any argument was provided
    if [ $# -eq 0 ]; then
        echo -e "${RED}[ERROR] No project type specified${NC}"
        print_help
        exit 1
    fi

    case "$1" in
        pytorch)
            get_pytorch_build_projects
            ;;
        sdk)
            get_sdk_build_projects
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown project type: $1${NC}"
            print_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
