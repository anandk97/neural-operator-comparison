import sys
from pathlib import Path

# neuraloperator's local-kernel (DISCO) layers need two pure-Python modules from torch-harmonics, which has no
# Windows wheels; a minimal copy lives in third_party/torch_harmonics_min.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "torch_harmonics_min"))
