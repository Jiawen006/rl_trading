import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utility.preprocessor import *
from utility.run_model import run_Model

if __name__ == "__main__":
    run_Model.run()
