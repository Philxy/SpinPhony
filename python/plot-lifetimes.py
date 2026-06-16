import pandas as pd
import matplotlib.pyplot as plt


filename = "Outputs/equilibrium_lifetimes.csv"

# header of file: "q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps"


# plot energy of magnons vs lifetime
df = pd.read_csv(filename)
plt.figure(figsize=(8, 6))
plt.scatter(df["energy_meV"], df["tau_ps"], s=10, alpha=0.7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy (meV)")
plt.ylabel("Lifetime (ps)")
plt.title("Magnon Lifetimes vs Energy")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("Outputs/lifetimes_vs_energy.png", dpi=300)
plt.show()