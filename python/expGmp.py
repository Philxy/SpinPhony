import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data_no_log = """
136.35335770556202, 17.81926701570682
217.61563410900393, 67.37801047120419
217.55586855772793, 74.00910994764399
248.76135021215023, 78.02267015706806
246.93063700990496, 81.1637172774869
279.28110080456315, 91.45937172774869
278.6205341851962, 98.09047120418847
308.6574420344054, 98.43947643979058
309.217350883202, 102.9765445026178
340.4448514249366, 104.54706806282722
340.9984691630725, 109.78214659685864
372.80789744091595, 113.44670157068063
372.1835047078478, 116.06424083769633
370.33391817362065, 121.29931937172776
372.10643860225497, 124.61486910994763
406.4118650347059, 117.98376963350783
402.72527418757295, 127.0579057591623
404.97434624875046, 144.15916230366494
342.2063624099149, 109.08413612565445
340.35992143101794, 113.97020942408376
340.43541475894557, 105.5940837696335
276.26608602045314, 92.68089005235602
278.0275970054313, 97.21795811518324
246.2810798341943, 86.5732984293194
247.57075751962472, 76.80115183246075
217.606197443013, 68.42502617801048
217.56530522371887, 72.96209424083767
161.41399702220758, 37.014554973822
161.9817697593301, 40.67910994764399
165.51422839527748, 48.706230366492164
190.0684333037418, 57.43136125654453
189.47549612397685, 56.55884816753929
188.25659343347849, 58.478376963350804
"""

# 1. Load and correct the data
temperature_ext, G_ext = np.loadtxt(data_no_log.strip().splitlines(), delimiter=',', unpack=True)
x1, x2 = 50.0, 500.0
temperature_true = x1 * (x2 / x1) ** ((temperature_ext - x1) / (x2 - x1))
y1, y2 = 0.02, 200.0
G_true = y1 * (y2 / y1) ** ((G_ext - y1) / (y2 - y1))
G_scaled = G_true * 1e15

# Sort the data for clean plotting of line fits
sort_idx = np.argsort(temperature_true)
T_sorted = temperature_true[sort_idx]
G_sorted = G_scaled[sort_idx]

# Convert to log10 space for linear fitting
log_T = np.log10(T_sorted)
log_G = np.log10(G_sorted)

# ==========================================
# FIT 1: Single Power Law
# ==========================================
# Fits a straight line in log-log space: log(G) = slope * log(T) + intercept
slope_single, intercept_single = np.polyfit(log_T, log_G, 1)
G_fit_single = (10**intercept_single) * (T_sorted**slope_single)

# ==========================================
# FIT 2: Broken (Piecewise) Power Law
# ==========================================
# Defines a continuous line that changes slope at a specific break point
def broken_power_law(x, x_break, y_break, slope1, slope2):
    return np.piecewise(
        x, 
        [x < x_break], 
        [lambda x: slope1 * (x - x_break) + y_break, 
         lambda x: slope2 * (x - x_break) + y_break]
    )

# Initial guesses for the curve_fit: [break_X, break_Y, slope1, slope2]
p0 = [np.median(log_T), np.median(log_G), 2.0, 2.0]

popt, _ = curve_fit(broken_power_law, log_T, log_G, p0=p0)
log_T_break, log_G_break, slope1, slope2 = popt

# Calculate the actual breakpoint temperature
T_break = 10**log_T_break
G_fit_broken = 10**(broken_power_law(log_T, *popt))

# ==========================================
# T^2 REFERENCE LINE
# ==========================================
T_ref = np.geomspace(min(T_sorted), max(T_sorted), 100)
# Anchor to the lowest temperature point (index 0 of sorted data)
coeff_T2 = G_sorted[0] / (T_sorted[0] ** 2)
G_ref_T2 = coeff_T2 * (T_ref ** 2)

# ==========================================
# PLOTTING
# ==========================================
plt.figure(figsize=(10, 7))

# Original Data
plt.loglog(T_sorted, G_sorted, 'o', markersize=7, color='steelblue', alpha=0.8, label='Experimental data')

# Single Fit Line
plt.loglog(T_sorted, G_fit_single, 'k--', alpha=0.6, 
           label=f'Single fit: $T^{{{slope_single:.2f}}}$')

# Broken Fit Line
plt.loglog(T_sorted, G_fit_broken, 'r-', linewidth=2, 
           label=f'Broken fit:\n$T^{{{slope1:.2f}}}$ (T < {T_break:.1f} K)\n$T^{{{slope2:.2f}}}$ (T > {T_break:.1f} K)')

# T^2 Reference Line
plt.loglog(T_ref, G_ref_T2, color='green', linestyle='-.', alpha=0.6, 
           label=r'$T^2$ Reference')

# Mark the breakpoint
plt.axvline(T_break, color='red', linestyle=':', alpha=0.5)

# Formatting
plt.xlabel('Temperature (K)', fontsize=12)
plt.ylabel(r'Coupling Constant, G ($W/m^3/K$)', fontsize=12)
plt.title('Power-Law Fits for Coupling Constant vs. Temperature', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=11)

# State material: Ca9La5Cu24O41 in latex formatting
plt.title( r"$G_{mp}$ vs Temperature for $\mathrm{Ca}_9\mathrm{La}_5\mathrm{Cu}_{24}\mathrm{O}_{41}$", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("Outputs/G_mp_power_law_fits.png", dpi=300)
plt.show()

# Print results to console
print(f"--- Single Power Law ---")
print(f"Exponent: {slope_single:.3f}")
print(f"\n--- Broken Power Law ---")
print(f"Break Temperature: {T_break:.2f} K")
print(f"Low-T Exponent:  {slope1:.3f}")
print(f"High-T Exponent: {slope2:.3f}")