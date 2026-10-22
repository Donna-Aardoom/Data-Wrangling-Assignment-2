# LOF: Identifying Density-Based Local Outliers

This code was written by Donna Aardoom for the individual assignment of the Data Wrangling course at Utrecht University.

It reconstructs the synthetic data experiment presented in Figure 9 of *Identifying Density-Based Local Outliers* by Breunig et al. (2000), in the "Experiments" section.

## Description

The experiment uses a synthetically generated 2D dataset consisting of Gaussian and uniform clusters with different densities, with six additional outliers. The Local Outlier Factor (LOF) algorithm is applied to examine if an object is has a high possibility of being an outlier, based on its local density.

The code uses `LocalOutlierFactor` from `scikit-learn`, with `n_neighbors=40`, which is referred as MinPts in the paper. The resulting LOF scores are visualized in a plot with the dataset on the left and a three-dimensional plot on the right.

**Note:** The original dataset Breunig et al. used was not available. The cluster parameters and outlier locations were estimated from the figure in thier paper. Therefore, this is a reconstruction of the experiment, not an exact replication.

## Requirements

- Python 3.11 or compatible
- NumPy
- Matplotlib
- scikit-learn

## How to run

Install the required packages:

```bash
python -m pip install numpy matplotlib scikit-learn
```

Run the script from the project directory:

```bash
python code.py
```

The script generates the synthetic dataset, calculates the LOF scores, and saves the visualization as `lof_figure9.png`.

## Reference

Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *Identifying Density-Based Local Outliers*. Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data.

https://doi.org/10.1145/335191.335388

