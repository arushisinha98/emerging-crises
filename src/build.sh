#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=06:00:00

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Initialize module system for PBS environment
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

# Load prerequisite modules in correct order
echo "Loading prerequisite modules..."
module load tools/prod || { echo "Failed to load tools/prod"; exit 1; }

echo "Loading GCCcore module..."
module load GCCcore/11.2.0 || { echo "Failed to load GCCcore"; exit 1; }

echo "Loading Python module..."
module load Python/3.9.6-GCCcore-11.2.0 || { echo "Failed to load Python module"; exit 1; }

# Verify modules are loaded
echo "Currently loaded modules:"
module list

# Verify Python is available
echo "Verifying Python installation..."
python3 --version || { echo "Python3 not available after module load"; exit 1; }
which python3 || { echo "Python3 not found in PATH"; exit 1; }

echo "Python module loaded successfully: $(python3 --version)"

# Set up working directories
WORK_DIR=${PBS_O_WORKDIR}
SCRATCH_DIR="/tmp/${USER}_${PBS_JOBID%.*}"

# Create scratch directory
mkdir -p ${SCRATCH_DIR} || {
    echo "Failed to create scratch directory, using work directory"
    SCRATCH_DIR="${WORK_DIR}/temp_${PBS_JOBID%.*}"
    mkdir -p ${SCRATCH_DIR}
}

echo "Created scratch directory: ${SCRATCH_DIR}"
cd ${WORK_DIR}

# =============================================================================
# ENVIRONMENT CREATION AND ACTIVATION
# =============================================================================

echo "Creating Python virtual environment..."

# Create virtual environment
python3 -m venv ${SCRATCH_DIR}/venv || {
    echo "Failed to create virtual environment"
    exit 1
}

# Activate virtual environment
source ${SCRATCH_DIR}/venv/bin/activate || {
    echo "Failed to activate virtual environment"
    exit 1
}

# Verify activation
which python || { echo "Virtual environment activation failed"; exit 1; }
python --version || { echo "Python not working in virtual environment"; exit 1; }

echo "Virtual environment activated successfully: $(python --version)"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Install requirements..."
python -m pip install -r requirements.txt

# =============================================================================
# COPY FILES TO SCRATCH DIRECTORY
# =============================================================================

echo "Copying project files to scratch directory..."

# Copy all Python files
cp *.py ${SCRATCH_DIR}/
cp config.json ${SCRATCH_DIR}/

# Copy data files if they exist
cp *.xlsx ${SCRATCH_DIR}/ 2>/dev/null || true

# Copy environment file if it exists
if [ -f .env ]; then
    cp .env ${SCRATCH_DIR}/
    echo "Environment file copied"
fi

# Change to scratch directory for execution
cd ${SCRATCH_DIR}

# =============================================================================
# JOB EXECUTION
# =============================================================================

echo "Starting main.py execution..."
echo "Job started at: $(date)"

# Set Python path to include current directory
export PYTHONPATH="${SCRATCH_DIR}:${PYTHONPATH}"

# Run the main Python script
python main.py

# Capture exit status
EXIT_STATUS=$?

echo "Job completed at: $(date)"
echo "Exit status: ${EXIT_STATUS}"

# =============================================================================
# RESULTS COLLECTION
# =============================================================================

echo "Collecting results..."

# Copy output files back to working directory
if [ -f worldbank_variables.csv ]; then
    cp worldbank_variables.csv ${WORK_DIR}/
    echo "Copied worldbank_variables.csv"
fi

if [ -f worldbank_variables_classified.csv ]; then
    cp worldbank_variables_classified.csv ${WORK_DIR}/
    echo "Copied worldbank_variables_classified.csv"
fi

# Copy log files
if [ -f log.txt ]; then
    cp log.txt ${WORK_DIR}/
    echo "Copied log.txt"
fi

# Copy any other output files that might have been generated
cp *.csv ${WORK_DIR}/ 2>/dev/null || true
cp *.txt ${WORK_DIR}/ 2>/dev/null || true

# =============================================================================
# CLEANUP
# =============================================================================

echo "Performing cleanup..."

# Deactivate environment
deactivate

# Change back to working directory
cd ${WORK_DIR}

# Remove scratch directory and all contents
rm -rf ${SCRATCH_DIR}
echo "Scratch directory and temporary environment cleaned up"

echo "PBS job completed successfully"

# Exit with the same status as the main script
exit ${EXIT_STATUS}