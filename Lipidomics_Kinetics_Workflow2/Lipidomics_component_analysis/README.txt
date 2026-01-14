# Lipid Component Analysis Pipeline

This repository contains a modular Python pipeline for estimating lipid component n-values from mass spectrometry data.  
It combines Monte Carlo simulation, bootstrap resampling, and linear algebra to propagate experimental uncertainty into robust component-level estimates.  

The workflow produces:  
- Component n-value tables with confidence intervals  
- Bar plots of component distributions per genotype  
- Conformity scatter plots comparing genotypes  
- Specialized comparisons (e.g., palmitate vs. stearate vs. literature)  



# Repository Structure

#   1. sampling.py       → draws asymmetric Gaussian samples, runs MC fits
#   2. stats_tests.py    → provides correlation-adjusted paired t-test
#   3. design_matrix.py  → builds the design matrix for lipid components
#   4. parsing.py        → extracts fatty acid tokens and structural groups
#   5. bootstrap.py      → runs bootstrap resampling with Monte Carlo fits
#   6. plotting.py       → makes bar plots and conformity scatter plots
#   7. main.py           → user interface (file dialogs), data prep, orchestration


# Workflow Overview

1. **Input**  
   Provide a CSV file containing lipid regression results with confidence intervals.  

2. **Parsing**  
   Lipids (from a few common classes—can easily be expanded) are split into fatty acid chains and structural groups (backbone and headgroups) (e.g., glycerol, choline).  

3. **Design Matrix Construction**  
   A linear system is built:  

   A * x ≈ b

   - A = design matrix (lipid × component counts)  
   - x = unknown component n-values  
   - b = observed lipid values  

4. **Monte Carlo Simulation**  
   - Lipid values are resampled from asymmetric Gaussian distributions.  
   - Non-negative least squares is solved repeatedly to obtain distributions of component values.  

5. **Bootstrap Resampling**  
   - Lipids are randomly dropped in each iteration (currently 1000 iterations with 5% dropped each time).
   - Monte Carlo simulation is re-run, aggregating across bootstraps.  
   - Outputs medians + 95% confidence intervals per component.  

6. **Statistical Testing**  
   A correlation-adjusted paired t-test compares component values across genotypes.  

7. **Plotting**  
   - Bar plots of component medians + CI  
   - Conformity scatter plots vs. baseline genotype  
   - Palmitate/stearate comparisons vs. literature  

8. **Output**  
   - SVG plots  
   - Component n-value tables (CSV)  


# Example Usage
**Assuming zip file is extracted to Documents**
1.Open command line
2."cd C:\...\Documents\Lipidomics_component_analysis"
3."python main.py"


Interactive flow:
1. Select input CSV file (lipid regression with CI).  
2. Save bar plot of component n-values.  
3. Select baseline genotype (Case sensative; in the example input file we used "APOE3")
4. Save conformity scatter plots.  
5. Save palmitate vs. stearate comparison plot.  

All figures are shown at the end.  


# Dependencies
- Python ≥ 3.10  
- NumPy, Pandas  
- SciPy  
- Matplotlib  
- Tkinter (for file dialogs)  


# Authorship
This code was authored by Coleman Nielsen, with support from ChatGPT.  
