# EBM

This project will initially be a Python port of the R package [EBM](https://github.com/donaldcummins/EBM) for maximum likelihood estimation of k-box stochastic energy balance models. In the future, it will serve as a base upon which to add new methodologies and features as and when they are developed.

## How to cite

Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal Estimation of Stochastic Energy Balance Model Parameters, *Journal of Climate, 33*(18), 7909-7926, [https://doi.org/10.1175/JCLI-D-19-0589.1](https://doi.org/10.1175/JCLI-D-19-0589.1)

## Quickstart

The easiest way to try out EBM is to clone this repository and build a fresh conda environment from the [YAML file](EBM.yml).

```bash
git clone git@github.com:cemac/EBM.git
cd EBM
conda env create -f EBM.yml
conda activate EBM
```

You can then import EBM as a Python module from within the interpreter.

```python
import energy_balance_model as ebm
```

The file [demo.py](demo.py) contains a script showing how to generate synthetic data from a three-box stochastic EBM and how to estimate the EBM's parameters via maximum likelihood.

## Licence

EBM is licenced under the MIT license - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Thanks to Chris Smith for providing ensembles of [calibrated parameter values](https://doi.org/10.5281/zenodo.7112539), which we use here for initialisation and (optionally) regularisation of the maximum likelihood estimation.

## References

Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal Estimation of Stochastic Energy Balance Model Parameters, *Journal of Climate, 33*(18), 7909-7926, [https://doi.org/10.1175/JCLI-D-19-0589.1](https://doi.org/10.1175/JCLI-D-19-0589.1)

Smith, C. (2024). FaIR calibration data (1.4.0) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.10566646](https://doi.org/10.5281/zenodo.10566646)

Smith, C. (2024). FaIR calibration data (1.4.3) [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.13951079](https://doi.org/10.5281/zenodo.13951079)