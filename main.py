import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utility import preprocessor, run_model

if __name__ == "__main__":
    run_model.run()
