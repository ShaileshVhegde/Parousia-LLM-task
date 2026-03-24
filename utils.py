import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    """
    Utility function to save matplotlib figures to a centralized plots directory.
    """
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filepath}")
