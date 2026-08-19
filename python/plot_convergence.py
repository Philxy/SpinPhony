import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots


plt.style.use("science")

# ==========================================
# CONFIGURATION
# ==========================================
base_dir = "Outputs"

# Define the k-points (q_idx) and branches you want to track
q_indices = [0]         # 0 is Gamma
branches = [i for i in range(22,23)]        # Mode 22 is good example at Gamma

# ==========================================

def main():
    # Dictionary to store the extracted data dynamically
    # Format: {(grid, q_idx, branch): [(sigma1, tau1), (sigma2, tau2), ...]}
    plot_data = {}

    # Update the search pattern to match the new directory structure
    search_pattern = os.path.join(base_dir, "CrI3_Path_Hyrbid_grid_*_sig_*_15K")
    directories = glob.glob(search_pattern)

    if not directories:
        print(f"No directories found matching: {search_pattern}")
        return

    print(f"Found {len(directories)} directories. Parsing CSV files...")

    for d in directories:
        # Extract both the grid size and sigma value using regex
        match = re.search(r'grid_([0-9]+)_sig_([0-9.]+)_', d)
        if not match:
            continue
        
        grid = int(match.group(1))
        sigma = float(match.group(2))
        filepath = os.path.join(d, "hybrid_path_lifetimes.csv")
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found in {d}")
            continue
            
        # Read the CSV. comment='#' ignores the path_labels header line natively
        df = pd.read_csv(filepath, comment='#')
        
        # Extract tau_ps for requested q_indices and branches
        for q in q_indices:
            for b in branches:
                row = df[(df['q_idx'] == q) & (df['branch'] == b)]
                if not row.empty:
                    tau = row['tau_ps'].values[0]
                    
                    # Create the dictionary key if it doesn't exist
                    key = (grid, q, b)
                    if key not in plot_data:
                        plot_data[key] = []
                    
                    plot_data[key].append((sigma, tau))

    # ==========================================
    # PLOTTING
    # ==========================================
    # plt.style.use('science') # Uncomment if you want to enforce the scienceplots style

    plt.figure(figsize=(8/2.5, 8/2.5))

    #colors = plt.cm.tab10.colors  
    markers = ['o', 's', '^', 'D', 'v', '<', '>'] # Add markers to distinguish grids better
    
    # Sort keys so the legend is organized neatly (by grid size, then q, then branch)
    sorted_keys = sorted(plot_data.keys(), key=lambda x: (x[0], x[1], x[2]))

    for idx, key in enumerate(sorted_keys):
        grid, q, b = key
        values = plot_data[key]
        
        # Sort values by sigma so the line plots correctly from left to right
        values.sort(key=lambda x: x[0])
        
        sigmas = [v[0] for v in values]
        taus = [v[1] for v in values]
        
        plt.plot(
            sigmas, taus, 
            marker=markers[idx % len(markers)], 
            linestyle='-', 
            linewidth=2,
            label=rf"${grid} \times {grid} \times {grid}$",
        )

    plt.xlabel('Smearing $\sigma$ (meV)', fontsize=12)
    plt.ylabel(r'Lifetime $\tau$ (ps)', fontsize=12)
    
    # Log scale is usually best for visualizing smearing convergence
    plt.xscale('log') 
    plt.yscale('log') # Uncomment if lifetimes span multiple orders of magnitude
    

    plt.legend(fontsize=11,title=r"$\mathbf{q}$-mesh", frameon=True)
    plt.tight_layout()
    
    # Save the figure and show it
    plt.savefig('Outputs/sigma_convergence_Gamma.png', dpi=300)
    print("Saved plot as 'sigma_convergence.png'")
    plt.show()

if __name__ == "__main__":
    main()