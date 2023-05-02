#!/bin/bash

# Clone dct-policy repository
git clone https://github.com/csquires/dct-policy.git
cd dct-policy

# Setup dct-policy environment
# Note: Given setup script has some missing pip installs
bash setup.sh
source venv/bin/activate
pip install seaborn tqdm ipdb p_tqdm
pip install networkx==2.8.8

# Grab PADS source files
wget -r --no-parent --no-host-directories --cut-dirs=1 http://www.ics.uci.edu/\~eppstein/PADS/

# Copy our files into dct-policy folder
cp ../our_code/pdag.py venv/lib/python3.8/site-packages/causaldag/classes/
cp ../our_code_baseline/*.py baseline_policies
cp ../our_code/*.py .

# Run experiments to obtain plots
python3 exp1_type1_alpha0_beta1.py
python3 exp2_type1_alpha0_beta1.py
python3 exp3_type1_alpha0_beta1.py
python3 exp4_type1_alpha0_beta1.py
python3 exp5_type1_alpha0_beta1.py

python3 exp1_type1_alpha1_beta1.py
python3 exp2_type1_alpha1_beta1.py
python3 exp3_type1_alpha1_beta1.py
python3 exp4_type1_alpha1_beta1.py
python3 exp5_type1_alpha1_beta1.py

python3 exp1_type2_alpha0_beta1.py
python3 exp2_type2_alpha0_beta1.py
python3 exp3_type2_alpha0_beta1.py
python3 exp4_type2_alpha0_beta1.py
python3 exp5_type2_alpha0_beta1.py

python3 exp1_type2_alpha1_beta1.py
python3 exp2_type2_alpha1_beta1.py
python3 exp3_type2_alpha1_beta1.py
python3 exp4_type2_alpha1_beta1.py
python3 exp5_type2_alpha1_beta1.py

# Return to parent directory
cd ..

