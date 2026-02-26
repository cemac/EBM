"""
Download and process calibrated EBM parameter ensembles.

This script downloads parameter ensembles from Chris Smith's calibrated
two-box and three-box energy balance models, computes log-space statistics
(means and standard deviations), and saves them in NumPy binary format.

The output file 'parameter_distributions.npz' contains:
- log_means_2, log_stds_2: Statistics for the two-box model (9 parameters)
- log_means_3, log_stds_3: Statistics for the three-box model (11 parameters)

Data sources:
- Two-box: https://doi.org/10.5281/zenodo.13951079
- Three-box: https://doi.org/10.5281/zenodo.10566646
"""

import numpy as np
import pandas as pd

URL2 = "https://zenodo.org/records/13951079/files/calibrated_constrained_parameters.csv?download=1"
URL3 = "https://zenodo.org/records/10566646/files/calibrated_constrained_parameters.csv?download=1"

# Chris Smith's two-box parameter ensemble
df2 = pd.read_csv(URL2, index_col=0)
df2 = df2.iloc[:, :9]  # drop non-EBM parameters
log_df2 = df2.apply(np.log)
log_means_2 = log_df2.mean().to_numpy()
log_stds_2 = log_df2.std().to_numpy()

# Chris Smith's three-box parameter ensemble
df3 = pd.read_csv(URL3, index_col=0)
df3 = df3.iloc[:, :11]  # drop non-EBM parameters
log_df3 = df3.apply(np.log)
log_means_3 = log_df3.mean().to_numpy()
log_stds_3 = log_df3.std().to_numpy()

# Save results in binary format
np.savez(
    "parameter_distributions.npz",
    log_means_2=log_means_2,
    log_stds_2=log_stds_2,
    log_means_3=log_means_3,
    log_stds_3=log_stds_3,
)
