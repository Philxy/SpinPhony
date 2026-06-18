import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "Outputs/equilibrium_lifetimes.csv"

# header of file: "q_idx,qx,qy,qz,particle,branch,energy_meV,gamma_ps-1,tau_ps"


literature_scattering_rates = """3.5330366949927345, 0.09102981779915223
7.1097094323124335, 0.21363397911526588
10.706058931542348, 0.3442640548857818
14.470842848333064, 0.4563947703262179
17.656864338394996, 0.5710470594624236
21.17999854447029, 0.6094377562050983
24.976458841488746, 0.7145014915214672
27.510987074215667, 0.8685113737513525
31.712798486156796, 0.8811603465430217
38.91557544867763, 0.9963917920135462
44.60491499417326, 1.2199505557502925
52.00551275510881, 1.2557439628235745
60.29022624209932, 1.6173023172161325
57.937928431613116, 1.2557439628235751
81.95590434970995, 1.7638750529078007
81.95590434971, 1.4199606034664953
101.72024731297093, 1.828800808389492
130.63194185325634, 1.6173023172161325
157.5908014514459, 1.5045063827385736
187.96373839101864, 1.5154225728627575
221.65575534665635, 1.815627230483809
259.9051079393094, 2.606228915488146
294.5342488814524, 3.5054273703936762
330.00347911252817, 4.923882631706749
369.74408456724194, 5.9421053233155
438.50576628007633, 9.785451025760969
547.358170107669, 7.119234215334258
491.3127387829667, 3.7682364621484377
494.1140015119873, 4.7490758412131004
514.1751827683922, 5.731149775014003
391.3745601980384, 7.877516562652843
245.5406814401623, 2.1596327552241537
318.9361179185746, 4.050748891445512
112.68130079648533, 1.3894954943731381
154.92569632334198, 1.3019661288117441
226.7543125870802, 1.5045063827385743
256.9665200810953, 1.8824579107057196
289.5532174388399, 2.389659095901388
320.7545602050025, 2.968434514256669
359.38136638046257, 3.6608276068155456
451.1500269578206, 7.543120063354623
414.27044478476745, 4.680903310144551
446.0491499417328, 5.9852192324297695
491.3127387829667, 7.708505252846041"""


# plot energy of magnons vs lifetime. Only lines with particle="magnon" should be plotted. Use log-log scale for better visualization.
df = pd.read_csv(filename)
df = df[df["particle"] == "magnon"]
plt.figure(figsize=(8, 6))

plt.scatter(
    df["energy_meV"], df["tau_ps"], s=10, alpha=0.7, label="Current work"
)


# plot the literature lifetimes in the same plot for comparison
literature_data = pd.DataFrame(
    [line.split(",") for line in literature_scattering_rates.split("\n")],
    columns=["energy_meV", "gamma_pps"],
)
literature_data["energy_meV"] = literature_data["energy_meV"].astype(float)
literature_data["gamma_pps"] = literature_data["gamma_pps"].astype(float)
tau_lit = 1.0 / literature_data["gamma_pps"].astype(float)

plt.scatter(
    literature_data["energy_meV"],
    tau_lit,
    s=10,
    alpha=0.7,
    color="red",
    label="Literature",
)


# --- ADD POWER LAW SCALING COMPARISON LINES ---
# Define an energy range to display the reference lines
E_min, E_max = 5.0, 300.0
E_line = np.logspace(np.log10(E_min), np.log10(E_max), 100)

# Anchor point to position lines visually alongside the data
E_ref = 10.0
tau_ref = 3.0

# E^-2 scaling line (corresponding to scattering rate gamma proportional to E^2)
tau_E2 = tau_ref * (E_line / E_ref) ** (-2)
plt.plot(
    E_line,
    tau_E2,
    label=r"$\tau \propto E^{-2}$ ($\gamma \propto E^2$)",
    color="black",
    linestyle="--",
    linewidth=1.5,
)

# E^-1 scaling line (corresponding to scattering rate gamma proportional to E^1)
tau_E1 = tau_ref * (E_line / E_ref) ** (-1)
plt.plot(
    E_line,
    tau_E1,
    label=r"$\tau \propto E^{-1}$ ($\gamma \propto E^1$)",
    color="blue",
    linestyle="-.",
    linewidth=1.5,
)
# ----------------------------------------------


plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy (meV)")
plt.ylabel("Lifetime (ps)")
plt.title("Magnon Lifetimes vs Energy")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Outputs/lifetimes_vs_energy.png", dpi=300)
plt.show()