# USF-HF
HF Python Code for GPU

Title: Supporting Code for "Differentiable Tensor Formulations for Quantum Chemistry Methods in JAX"

Authors: Andrei Pomorov, Sagar Pandit

This archive contains two code files used to generate the results and figures presented in the manuscript submitted to the American Chemical Society (ACS) journal.


CONTENTS
--------

1. usf_hf.py
   - Description: A self-contained executable Python script that performs Hartree–Fock energy optimization via a hybrid method combining a modified secant phase and a trust-region constrained phase. It benchmarks convergence time and SCF energy for a given molecular geometry using PySCF and JAX.
   - Usage: Run with Python 3.10+ in an environment with PySCF, JAX, NumPy, SciPy, and Matplotlib installed.
   - Purpose: Demonstrates the timing, energy convergence, and optimization performance of the proposed differentiable SCF framework.

2. usf_hf_plots_code.ipynb
   - Description: A Jupyter notebook for generating the convergence plots shown in the manuscript. This includes constraint norm reduction across both optimization phases and total SCF energy convergence curves.
   - Requirements: Same environment as above with Jupyter Notebook.
   - Purpose: Produces visual figures for supporting convergence analysis (Figures 1a–b and 2a-b in manuscript).

INSTRUCTIONS
------------

To execute the main optimizer script:

    python usf_hf.py

This will print out energy values, constraint norm, and runtime statistics to stdout.

To generate the figures:

    1. Open `usf_hf_plots_code.ipynb` in Jupyter Notebook
    2. Ensure the `results` dictionary is preloaded from the previous run or re-run the optimization portion if needed
    3. Execute the notebook to view and save the plots

DEPENDENCIES
------------

The following Python packages are required:

- Python 3.10+
- JAX (with GPU support)
- NumPy
- SciPy
- Matplotlib
- PySCF

COMMENTS
--------

- The molecular geometries are set in the `mol.atom` list. You may uncomment alternative geometries as needed.
- Timing comparisons are averaged over multiple runs using randomized initial orbital guesses.
- All code is intended for research and academic reproducibility purposes only.

CONTACT
-------

For questions, please contact:

Andrei Pomorov 
apomorov@usf.edu  

