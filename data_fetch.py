"""Download the benchmark files into data/ (about 5 GB unpacked).

These are the FNO and Geo-FNO benchmark files (Li et al., ICLR 2021; Li et al., JMLR 2023) that Transolver and
Transolver++ also report on, from the Google Drive folders linked in
github.com/thuml/Transolver/tree/main/PDE-Solving-StandardBenchmark. The advection task is generated in
noc/data.py from its exact solution and needs no download.

    uv run python data_fetch.py
"""

import zipfile
from pathlib import Path

import gdown

DATA = Path(__file__).parent / "data"
GDRIVE = {
    "Burgers_R10.zip": "16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe",
    "Darcy_421.zip": "1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf",
    "NavierStokes_V1e-5_N1200_T20.zip": "1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5",
    "airfoil/NACA_Cylinder_X.npy": "16EY0obqsccypaDFVY0wlsXX73SlD5TMy",
    "airfoil/NACA_Cylinder_Y.npy": "1rJUPtIhTAsG8TQnqV5mljjgQlyz0NJvJ",
    "airfoil/NACA_Cylinder_Q.npy": "1AjW0t0YolY680J6xTQJ_g5bqTSJAZDZc",
    "elasticity/Random_UnitCell_sigma_10.npy": "1Ia5izgUum-IQLdO6PW70HO8AdAqA_IVb",
    "elasticity/Random_UnitCell_XY_10.npy": "1I-fO-RsFvD3nqBuFrg67R0yqTFdD_gpA",
}

if __name__ == "__main__":
    for rel, fid in GDRIVE.items():
        dest = DATA / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=fid, output=str(dest), quiet=False)
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as z:
                z.extractall(DATA)
