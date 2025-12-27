#!/bin/bash
# =============================================================================
# Customer Analytics ML Pipeline - Setup Script
# =============================================================================
# This script sets up the development environment for the project.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Options:
#   --no-venv    Skip virtual environment creation
#   --dev        Install development dependencies
#   --help       Show this help message
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
CREATE_VENV=true
INSTALL_DEV=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            CREATE_VENV=false
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --help)
            echo "Customer Analytics ML Pipeline - Setup Script"
            echo ""
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-venv    Skip virtual environment creation"
            echo "  --dev        Install development dependencies"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Customer Analytics ML Pipeline Setup${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
if [ "$CREATE_VENV" = true ]; then
    echo ""
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}  Virtual environment already exists. Skipping creation.${NC}"
    else
        python3 -m venv .venv
        echo -e "${GREEN}✓ Virtual environment created at .venv${NC}"
    fi
    
    # Activate virtual environment
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"
python3 -m pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Install development dependencies
if [ "$INSTALL_DEV" = true ]; then
    echo ""
    echo -e "${YELLOW}Installing development dependencies...${NC}"
    python3 -m pip install pytest pytest-cov black isort flake8 mypy --quiet
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Validate installation
echo ""
echo -e "${YELLOW}Validating installation...${NC}"

# Test imports
python3 -c "
import numpy
import pandas
import sklearn
print('  ✓ Core packages imported successfully')

from src.config import RANDOM_STATE
print('  ✓ Configuration loaded')

from src.preprocessing import MedianImputer, IQRCapper
print('  ✓ Preprocessing module loaded')

from src.models import get_regression_models
print('  ✓ Models module loaded')

from src.evaluation import ModelLogger
print('  ✓ Evaluation module loaded')
"

# Check data file
echo -e "${YELLOW}Checking data file...${NC}"
if [ -f "Data/marketing_campaign.csv" ]; then
    echo -e "${GREEN}✓ Data file found${NC}"
else
    echo -e "${YELLOW}⚠ Data file not found at Data/marketing_campaign.csv${NC}"
    echo -e "${YELLOW}  Please ensure the dataset is in place before running notebooks.${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}=============================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo -e "Next steps:"
if [ "$CREATE_VENV" = true ]; then
    echo -e "  1. Activate the environment: ${YELLOW}source .venv/bin/activate${NC}"
    echo -e "  2. Run tests: ${YELLOW}make test${NC}"
    echo -e "  3. Start Jupyter: ${YELLOW}jupyter notebook${NC}"
else
    echo -e "  1. Run tests: ${YELLOW}make test${NC}"
    echo -e "  2. Start Jupyter: ${YELLOW}jupyter notebook${NC}"
fi
echo ""
echo -e "For development:"
echo -e "  - Format code: ${YELLOW}make format${NC}"
echo -e "  - Run linting: ${YELLOW}make lint${NC}"
echo -e "  - Run all checks: ${YELLOW}make check${NC}"
echo ""
