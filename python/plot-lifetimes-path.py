import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection

def plot_colored_line_dispersions(csv_file, output_image='lifetime_lineplots.png'):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # 1. Clean Numerical Noise and Calculate Tau
    # Replace negative or zero gammas (numerical noise at Gamma) with a tiny finite number
    df.loc[df['gamma_ps-1'] <= 0, 'gamma_ps-1'] = 1e-12
    df['tau_ps'] = 1.0 / df['gamma_ps-1']
    
    # Find realistic bounds for the color scale (ignoring the artificial 1e12 lifetimes)
    finite_taus = df.loc[np.isfinite(df['tau_ps']) & (df['tau_ps'] < 1e10), 'tau_ps']
    if not finite_taus.empty:
        tau_max = finite_taus.quantile(0.95) # Cap at 95th percentile
        tau_min = finite_taus.min()
    else:
        tau_max = 1e4
        tau_min = 1e-2
        
    # Cap the extreme outliers for the colormap
    df.loc[df['tau_ps'] > tau_max, 'tau_ps'] = tau_max

    # 2. Reconstruct the X-axis (Cumulative distance)
    q_points = df[['q_idx', 'qx', 'qy', 'qz']].drop_duplicates().sort_values('q_idx')
    dq = np.diff(q_points[['qx', 'qy', 'qz']].values, axis=0)
    dist = np.linalg.norm(dq, axis=1)
    cumulative_dist = np.insert(np.cumsum(dist), 0, 0.0)
    
    dist_map = dict(zip(q_points['q_idx'], cumulative_dist))
    df['k_dist'] = df['q_idx'].map(dist_map)

    # 3. Setup the Figure (Two subplots side-by-side)
    print("Generating lineplots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Shared Logarithmic Normalization for the colormap
    norm = LogNorm(vmin=tau_min, vmax=tau_max)
    cmap = 'viridis_r' # reversed: dark/purple = short lifetime, bright/yellow = long lifetime

    def plot_particle_lines(ax, particle_name, title):
        particle_df = df[df['particle'] == particle_name]
        branches = particle_df['branch'].unique()
        
        lc_list = []
        for b in branches:
            branch_data = particle_df[particle_df['branch'] == b].sort_values('q_idx')
            
            x = branch_data['k_dist'].values
            y = branch_data['energy_meV'].values
            z = branch_data['tau_ps'].values
            
            # Create line segments
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color each segment by the average lifetime of its two endpoints
            z_seg = (z[:-1] + z[1:]) / 2.0
            
            # Create and add the LineCollection
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(z_seg)
            lc.set_linewidth(2.5)
            ax.add_collection(lc)
            lc_list.append(lc)
            
        # Formatting for the specific axis
        ax.set_xlim(0, cumulative_dist[-1])
        # Auto-scale y-axis based on data
        if not particle_df.empty:
            ax.set_ylim(0, particle_df['energy_meV'].max() * 1.05)
        
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Wavevector Path Distance', fontsize=13)
        ax.grid(True, axis='both', linestyle=':', color='gray', alpha=0.5)
        
        return lc_list[0] if lc_list else None

    # Plot on respective axes
    lc_phon = plot_particle_lines(ax1, 'phonon', 'Phonon Dispersion')
    lc_mag = plot_particle_lines(ax2, 'magnon', 'Magnon Dispersion')
    
    ax1.set_ylabel('Energy (meV)', fontsize=14, fontweight='bold')

    # 4. Add a shared Colorbar
    # Use the mappable from one of the LineCollections
    mappable = lc_phon if lc_phon else lc_mag
    if mappable:
        # Adjust layout to make room for colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Lifetime $\\tau$ (ps)', fontsize=14, fontweight='bold')

    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"-> Saved plot to '{output_image}'")

if __name__ == "__main__":
    plot_colored_line_dispersions('Outputs/path_lifetimes.csv', 'Outputs/lifetime_colored_lines.png')