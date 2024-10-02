import energy_balance_model as ebm
import numpy as np
import matplotlib.pyplot as plt


# Create an instance of the EnergyBalanceModel class with random parameters
parameters = ebm.unstandardise(np.random.randn(11))
three_box_model = ebm.EnergyBalanceModel(*ebm.unpack_parameters(parameters))
print(three_box_model.get_parameters('dict'))

# Generate synthetic data
y = three_box_model.observe_noisy_step_response(150)
print(y[:5]) # print first five years of the data

# Estimate the parameters
estimation_results = ebm.fit_ebm(y, method='BFGS', options={'gtol': 1e-3})
fitted_model = estimation_results.get_model()
print(fitted_model.get_parameters('dict'))

# Compute the fitted values
fitted_model = estimation_results.get_model()
fitted_step_response = fitted_model.step_response(150)
fitted_values = fitted_model.observe(fitted_step_response)

def plot(y, fitted_values):
    """Plot the observed vs fitted values in two subplots."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    
    axes[0].plot(y[:,0], label='Observed')
    axes[0].plot(fitted_values[:,0], label='Fitted', linestyle='--')
    axes[0].set_ylim(0)
    axes[0].set_ylabel('Temperature anomaly (K)')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Observed vs fitted values')
    
    axes[1].plot(y[:,1], label='Observed')
    axes[1].plot(fitted_values[:,1], label='Fitted', linestyle='--')
    axes[1].set_ylim(0)
    axes[1].set_ylabel('Net radiative flux (W/m^2)')
    axes[1].set_xlabel('Time (years)')
    axes[1].legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('fitted_values.pdf')

# Plot the observed vs fitted values
plot(y, fitted_values)