#!/bin/bash

WEBAPP_DIR="/home/ubuntu/app/webapp"
VENV_DIR="/home/ubuntu/app/services/env"
MICROSERVICE_DIR="/home/ubuntu/app/services"

# Start the web app
echo "Starting web app on port 9000, serving files from $WEBAPP_DIR..."
cd "$WEBAPP_DIR" || { echo "Web app directory not found! Exiting."; exit 1; }
python3 -m http.server 9000 &
WEBAPP_PID=$!

# Activate venv for the microservice
echo "Activating virtual environment for microservice..."
source "$VENV_DIR/bin/activate" || { echo "Virtual environment activation failed! Exiting."; >

# Start microservice with gunicorn
echo "Starting microservice on port 8000 from $MICROSERVICE_DIR..."
cd "$MICROSERVICE_DIR" || { echo "Microservice directory not found! Exiting."; kill $WEBAPP_P>
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
