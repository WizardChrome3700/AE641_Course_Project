#!/bin/bash

python -m venv pursuit_guidance
source ./pursuit_guidance/bin/activate
pip3 install numpy==1.24.3 scipy==1.10.1 pandas==1.5.3 matplotlib
mkdir ./Outputs