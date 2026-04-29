import matplotlib.pyplot as plt
import numpy as np

def plot_equivalent_ksyn(m):
    """
    Plot the line of equivalent ksyn for a given m value.
    
    Parameters:
        m (float): Exponent in k_syn equation.
    """
    # Generate abundance log2FC values
    x = np.linspace(-2, 2, 100)  # log2FC abundance range
    y = (m - 1) * x              # corresponding fractional turnover log2FC
    
    # Create plot
    plt.figure(figsize=(6,6))
    plt.plot(x, y, label=f'Equivalent ksyn or kdeg line (m={m})', color='blue', linewidth=2)
    
    # Reference lines for context
    plt.axhline(0, color='gray', linestyle='--', label='y = 0 (constant k_frac)')
    plt.axvline(0, color='gray', linestyle='--', label='x = 0 (no abundance change)')
    plt.plot(x, -x, color='red', linestyle='--', label='y = -x (flux homeostasis)')
    
    # Labels and legend
    plt.xlabel('log2FC(Abundance)')
    plt.ylabel('log2FC(Fractional Turnover)')
    plt.title('Line of Equivalent ksyn or kdeg')
    plt.legend()
    plt.grid(True)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.show()

# Example usage:
m_value = float(input("Enter an order value for ksyn or kdeg: "))
plot_equivalent_ksyn(m_value)