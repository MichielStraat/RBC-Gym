#!/bin/bash

export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true

# Start fake X server if not running
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
sleep 3

