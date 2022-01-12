# call-timing
Code and data to run simulations and recreate figures from Norton et al. (2022)

Requires Python 3.7 and the packages listed in requirements.txt.

## Installation using Linux and pip
1. If not already installed, get the latest version of Python 3.7 (will not run in 3.8) through your package manager or download from https://www.python.org/downloads/
2. Clone this repository and navigate to the main directory of your local copy (the directory containing run_all.py)
3. Create a python environment (e.g. using pip):

`python3.7 -m venv ctm_env`

4. Activate the environment:

`source ctm_env/bin/activate`

5. Update some packages:

`pip install -U pip setuptools~=57.0 wheel`

6. Install required packages:

`pip install -r requirements.txt`

7. To reproduce all simulations and figures run the script run_all.py (make sure the environment is active):

`python3 run_all.py`

8. Figures will be located in the subdirectory fig/figures

Depending on your hardware, the whole simulation and figure generation process will take several hours.
If you have a multi-core processor and sufficient ram (~32GB), you can speed up the process by activating parallel processing (set b_multiprocessing to True in run_all.py, ~line 18).
If you want to only redo figure generation after having successfully run all simulations, you can set b_run_simulations to False in run_all.py, ~line 15.

The text in the figures will be slightly different, because the published figures were created using Latex (specifically lualatex) and font Helvetica. If you have lualatex installed, you can set B_USE_LATEX to True in pub/fig_ms.py, ~line 5.

If you have questions or problems, create an issue here or write to: philipp.norton@hu-berlin.de
