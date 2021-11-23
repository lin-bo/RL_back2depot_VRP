#!/usr/bin/env python
# coding: utf-8

# to get the right reqs file: pipreqs ./spo

# global vars
VENVS_DIR="PATH_TO_VENV"
VENV_NAME="rl_vrp"
GRB_VER="9.1.2"
LOGDIR="./logs"

# load module
echo "Load module..."
module load python/3.8
module load gurobi/$GRB_VER
# check if the license is set
gurobi_cl 1> /dev/null && echo Success || echo Fail
echo ""

# create virtual env
if [ ! -d "./$VENVS_DIR/$VENV_NAME" ]; then
  echo "Create venv..."
  # create source
  virtualenv --no-download $VENVS_DIR/$VENV_NAME
  source $VENVS_DIR/$VENV_NAME/bin/activate
  echo ""

  echo "Install requirements..."

  # install gurobipy
  echo "  Install GurobiPy..."
  cp -r $GUROBI_HOME/ .
  cd $GRB_VER
  python setup.py install
  cd ..
  rm -r $GRB_VER

  # pip install
  echo "  Install requirements..."
  pip install --no-index --upgrade pip
  pip install tqdm
  pip install numpy
  pip install pandas
  pip install matplotlib
  pip install ortools
  pip install torch==1.7.0
  pip install scipy
  pip install submitit

# activate virtual env
else
  echo "Activate venv..."
  source $VENVS_DIR/$VENV_NAME/bin/activate

fi
echo ""

# run . create_env.sh
