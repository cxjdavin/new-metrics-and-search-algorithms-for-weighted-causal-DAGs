#!/bin/bash

cd dct-policy
source venv/bin/activate

python3 exp5_type1_alpha0_beta1.py
python3 exp5_type1_alpha1_beta1.py
python3 exp5_type2_alpha0_beta1.py
python3 exp5_type2_alpha1_beta1.py

cd ..

