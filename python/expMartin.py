import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Raw data string for dataMartin
dataMartin = """
-0.4594432429514016, 0.009701023481915705
-0.3885502863039655, -0.0012356758882239305
-0.2999239035541232, -0.0011134326016737361
-0.16994669174012403, -0.011968636447447167
0.008972657232880769, -1.755170127000588
0.04410130301156549, -2.190983742661583
0.07037953484392043, -2.6102578416521593
0.09370762869416666, -3.024018768419262
0.11115174568506601, -3.404684362740654
0.13756037035641042, -3.6474065577411072
0.16988557009684763, -3.879086109189741
0.20515275826695645, -4.1273133268612305
0.28780959385686705, -4.209957938122486
0.33494660515112823, -4.386444645693114
0.4264416303587127, -4.502180514623863
0.4884434252976365, -4.551750167320499
0.6096639430183289, -4.419169173502953
0.6657287890070684, -4.507367704749866
0.7246948756632969, -4.667286372216545
0.8102855501304436, -4.777513143700039
0.9254590999430549, -4.832526697424437
0.9992940450201653, -4.860011063017433
1.105682377305941, -4.81020914807635
1.1735722238804298, -4.887356886219005
1.2562086855892485, -4.997587732478718
1.3300395558901397, -5.030589345071409
1.4246151119188224, -4.9752864822355045
1.4925131080457474, -5.04139972637878
1.5663399035704209, -5.079918585971158
1.6667994364584486, -5.05771105558097
1.7435722951890131, -5.1072603343965115
1.7789535770932379, -5.201004636076645
-0.008451085876928477, -1.3469182976807388
-0.04364492807510615, -0.9993806340148064
-0.064067706481646, -0.6518225964677802
-0.08160146854934913, -0.392536436139598
-0.10805491575909597, -0.21050395813574863
1.9503549639433242, -5.123526841060297
2.080222156799426, -5.283347713897743
2.186590115204111, -5.261132033955108
2.236864704186527, -5.189338551763417
2.3668215421194336, -5.22777999060764
2.452469263453638, -5.260765304095458
2.5262716103210003, -5.332387645685988
2.576419881240647, -5.431628820484713
2.6414899826720144, -5.3267114824137805
2.738999377577932, -5.298990779800114
2.818750897724136, -5.3154325018412925
2.8629988926795624, -5.403647332193076
2.9486792122235133, -5.392494669683362
3.031360496470734, -5.442035798946469
3.1584038694074974, -5.425308842570004
3.223445447405335, -5.359012233496906
3.3120107085119024, -5.441648695205721
3.3769341179994066, -5.535352249123672
3.459672449113685, -5.507651920391099
3.483346898942494, -5.452446852184446
3.568974246395606, -5.5130184006707115
3.6428784626684276, -5.44670956726896
3.7049087810408814, -5.457658490967747
3.746222937119071, -5.518291161097305
3.81711181899029, -5.534745107467128
3.8880455233999114, -5.4905093368403675
4.032806023334206, -5.484792425805972
4.109546283855024, -5.578479680619038
4.245415622080804, -5.611395722911145
1.8764507476705008, -5.189835674462053
0.3261573128480745, -4.287146424027326
0.7099767839624959, -4.595582535101651
0.8782120696908977, -4.805005658845474
2.157003165082428, -5.321862498713901
2.030033138117596, -5.239279009095921
"""

# --- 1. Load and sort data by time ---
raw_data = np.loadtxt(io.StringIO(dataMartin), delimiter=",")
data_sorted = raw_data[raw_data[:, 0].argsort()]
t_data = data_sorted[:, 0]
m_data = data_sorted[:, 1]


# --- 2. Type-II Demagnetization Model ---
def demag_model(t, t0, M0, A1, tau1, A2, tau2):
    dt = np.maximum(0, t - t0)
    step = np.heaviside(t - t0, 1.0)
    return M0 - step * (
        A1 * (1.0 - np.exp(-dt / tau1)) + A2 * (1.0 - np.exp(-dt / tau2))
    )


# --- 3. Perform Curve Fitting (UPDATED P0 & BOUNDS) ---
# Parameters: [t0, M0, A1, tau1, A2, tau2]
p0 = [-0.10, 0.0, 4.0, 0.15, 1.5, 1.5]
bounds = (
    [-0.5, -0.1, 0.0, 0.01, 0.0, 0.05],  # Lower bounds
    [0.5, 0.1, 10.0, 2.0, 10.0, 50.0],  # Upper bounds (A1 & A2 expanded to 10.0)
)

popt, pcov = curve_fit(demag_model, t_data, m_data, p0=p0, bounds=bounds)
perr = np.sqrt(np.diag(pcov))

t0_fit, M0_fit, A1_fit, tau1_fit, A2_fit, tau2_fit = popt

# --- 4. Calculate G_sl in SI Units (W / m^3 K) ---
C_s = 2.0e4  # Spin heat capacity at 15 K
C_l = 8.0e4  # Lattice heat capacity at 15 K
C_eff = (C_s * C_l) / (C_s + C_l)

tau2_sec = tau2_fit * 1e-12
tau2_err_sec = perr[5] * 1e-12

G_sl = C_eff / tau2_sec
G_sl_err = G_sl * (perr[5] / tau2_fit)

# --- 5. Print Results ---
print("=" * 55)
print("  FIT RESULTS: TYPE-II DEMAGNETIZATION & G_SL")
print("=" * 55)
print(f"Time-zero (t0)       : {t0_fit:8.4f} +/- {perr[0]:.4f} ps")
print(f"Baseline (M0)        : {M0_fit:8.4f} +/- {perr[1]:.4f}")
print(f"Fast Amplitude (A1)  : {A1_fit:8.4f} +/- {perr[2]:.4f}")
print(f"tau_1 (Fast drop)    : {tau1_fit:8.4f} +/- {perr[3]:.4f} ps")
print(f"Slow Amplitude (A2)  : {A2_fit:8.4f} +/- {perr[4]:.4f}")
print(f"tau_2 (Spin-Lattice)  : {tau2_fit:8.4f} +/- {perr[5]:.4f} ps")
print("-" * 55)
print(f"Effective C_eff      : {C_eff:.2e} J/(m^3 K)")
print(f"G_sl (SI Units)      : {G_sl:.3e} +/- {G_sl_err:.3e} W/(m^3 K)")
print("=" * 55)

# --- 6. Plot Data and Fit ---
t_dense = np.linspace(np.min(t_data), np.max(t_data), 1000)
m_fit = demag_model(t_dense, *popt)

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

ax.scatter(
    t_data,
    m_data,
    color="#1f77b4",
    s=25,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.5,
    label="Experimental Data",
)
ax.plot(
    t_dense,
    m_fit,
    color="#d62728",
    linewidth=2,
    label=(
        f"Type-II Fit\n"
        f"$\\tau_1 = {tau1_fit:.2f}\\text{{ ps}}$\n"
        f"$\\tau_2 = {tau2_fit:.2f}\\text{{ ps}}$\n"
        f"$G_\\text{{sl}} = {G_sl:.2e}\\text{{ W/(m}}^3\\text{{K)}}$"
    ),
)

ax.set_xlabel("Time (ps)", fontsize=12)
ax.set_ylabel("Magnetization $M(t) / M_0$", fontsize=12)
ax.set_title(
    "Ultrafast Demagnetization & Spin-Lattice Coupling ($CrI_3$)",
    fontsize=13,
    pad=10,
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(frameon=True, loc="lower right", fontsize=10)

# Inset zoom for ultrafast demagnetization region
ax_inset = fig.add_axes([0.45, 0.45, 0.38, 0.38])
ax_inset.scatter(
    t_data,
    m_data,
    color="#1f77b4",
    s=20,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.5,
)
t_inset = np.linspace(-0.2, 1.0, 300)
ax_inset.plot(t_inset, demag_model(t_inset, *popt), color="#d62728", linewidth=2)
ax_inset.set_xlim(-0.2, 1.0)
ax_inset.set_ylim(-5.2, 0.2)
ax_inset.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()