# PGland segmentation

## Python Env. Management
If you're using Conda to maintain your python environment, use the following steps to setup the environment for this project:
1. conda create --name {name of your env} --file requirements.txt
2. conda actiavte {name of your env}


If you're using Poetry instead, use these steps:
1. poetry install
2. poetry shell
3. python run train.py