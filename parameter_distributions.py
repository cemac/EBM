import numpy as np
import pandas as pd

# https://doi.org/10.5281/zenodo.13951079
URL2 = "https://zenodo.org/records/13951079/files/calibrated_constrained_parameters.csv?download=1"

# https://doi.org/10.5281/zenodo.10566646
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
