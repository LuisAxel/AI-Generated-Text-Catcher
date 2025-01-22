#!/bin/bash

# Get absolute paths
SCRIPT_DIR=$(dirname "$(realpath "$0")")

WEBAPP_DIR="$SCRIPT_DIR/webapp"
MICROSERVICE_DIR="$SCRIPT_DIR/services"

VENV_DIR="$MICROSERVICE_DIR/env"
REQUIREMENTS_FILE="$MICROSERVICE_DIR/requirements.txt"

# Start the web app
echo "Starting web app on port 9000, serving files from $WEBAPP_DIR..."
cd "$WEBAPP_DIR" || { echo "Web app directory not found! Exiting."; exit 1; }
python3 -m http.server 9000 &
WEBAPP_PID=$!

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found! Creating it..."
    cd "$MICROSERVICE_DIR"
    python3 -m venv env || { echo "Failed to create virtual environment! Exiting."; kill $WEBAPP_PID; exit 1; }

    # Activate the virtual environment
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate" || { echo "Virtual environment activation failed! Exiting."; kill $WEBAPP_PID; exit 1; }

    # Check if requirements.txt exists
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "requirements.txt file not found in $MICROSERVICE_DIR! Exiting."
        kill $WEBAPP_PID
        deactivate
        exit 1
    fi

    # Install the required dependencies for the microservice
    echo "Installing dependencies from $REQUIREMENTS_FILE..."
    pip install -r "$REQUIREMENTS_FILE" || { echo "Failed to install dependencies! Exiting."; kill $WEBAPP_PID; deactivate; exit 1; }
else
    echo "Activating existing virtual environment for microservice..."
    source "$VENV_DIR/bin/activate" || { echo "Virtual environment activation failed! Exiting."; kill $WEBAPP_PID; exit 1; }
fi

# Start microservice with flask
echo "Starting microservice on port 8000 from $MICROSERVICE_DIR..."
cd "$MICROSERVICE_DIR" || { echo "Microservice directory not found! Exiting."; kill $WEBAPP_PID; deactivate; exit 1; }
python3 predict_api.py &
MICROSERVICE_PID=$!

# Function to stop both services on script exit
cleanup() {
    echo "Stopping web app and microservice..."
    kill $WEBAPP_PID $MICROSERVICE_PID
    deactivate
    exit
}

# Trap SIGINT or SIGTERM and run cleanup
trap cleanup SIGINT SIGTERM

# Wait for both processes to complete
wait $WEBAPP_PID $MICROSERVICE_PID
