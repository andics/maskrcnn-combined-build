import os
import sys
from pathlib import Path
#[sys.path.remove(i) for i in [*(sys.path)[7:]]]

#This line may need modification depending on the location of this notebook. The task is to add .../maskrcnn_combined
module_path = str(Path(os.path.dirname(os.path.realpath("__file__"))).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)
print(f"Appended: {module_path}")
print(f"Current Python PATH: {sys.path}")