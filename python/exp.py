import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Raw data string from user
data = """
-14.232209737827773, -0.0008683068017365791
-10.674157303370825, -0.0008683068017365791
-7.6779026217229, 0.0002894356005788967
-5.243445692883961, -0.0008683068017365791
-0.18726591760303127, 0.00028943560057886897
-0.18726591760303155, -0.058755426917510845
-0.1872659176030421, -0.11201157742402312
-0.1872659176030428, -0.16526772793053549
-1.8117433919970204e-14, -0.18958031837916062
-0.18726591760305325, -0.22315484804630972
-0.1872659176030346, -0.27062228654124465
0.18726591760297637, -0.29956584659913177
59.737827715355785, -0.8888567293777137
61.985018726591754, -0.8992764109985532
64.2322097378277, -0.9293777134587557
66.29213483146066, -0.9316931982633865
69.1011235955056, -0.9351664254703331
71.53558052434455, -0.9467438494934879
74.3445692883895, -0.9479015918958034
76.21722846441946, -0.9467438494934881
77.90262172284643, -0.9617945007235893
81.27340823970036, -0.9617945007235893
84.4569288389513, -0.9652677279305355
87.82771535580524, -0.9698986975397976
92.88389513108613, -0.9745296671490593
0.936329588014951, -0.38755426917510866
3.1835205992509277, -0.4639652677279306
5.805243445692873, -0.4975397973950797
9.550561797752787, -0.5345875542691751
12.921348314606716, -0.5646888567293777
15.917602996254674, -0.5982633863965269
19.85018726591759, -0.6376266280752534
23.408239700374516, -0.6619392185238785
26.217228464419456, -0.6920405209840812
29.5880149812734, -0.710564399421129
31.64794007490637, -0.7290882778581766
33.707865168539314, -0.7395079594790162
36.70411985018725, -0.7591895803183792
38.014981273408225, -0.7696092619392187
42.32209737827716, -0.8008683068017368
40.07490636704118, -0.7846599131693202
49.625468164794, -0.8541244573082492
45.318352059925076, -0.8298118668596237
54.11985018726591, -0.8738060781476125
56.55430711610485, -0.8888567293777139
1.3108614232209557, -0.40955137481910275
2.2471910112359312, -0.43502170767004344
0.3745318352059765, -0.31461649782923307
0.3745318352059555, -0.3342981186685964
0.7490636704119515, -0.3539797395079596
1.6853932584269298, -0.37134587554269183
27.340823970037437, -0.6642547033285096
21.34831460674155, -0.6109985528219973
29.5880149812734, -0.6885672937771348
16.292134831460665, -0.5704775687409551
22.659176029962534, -0.6410998552821998
10.674157303370771, -0.5102749638205502
14.981273408239694, -0.5496382054992764
12.172284644194738, -0.5299565846599132
"""

dataMartin= """

"""

# --- 1. Load and sort data by time ---
raw_data = np.loadtxt(io.StringIO(data), delimiter=",")
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


# --- 3. Perform Curve Fitting ---
p0 = [-0.18, 0.0, 0.35, 0.8, 0.60, 25.0]
bounds = (
    [-2.0, -0.05, 0.0, 0.05, 0.0, 2.0],
    [2.0, 0.05, 1.0, 5.0, 1.5, 200.0],
)

popt, pcov = curve_fit(demag_model, t_data, m_data, p0=p0, bounds=bounds)
perr = np.sqrt(np.diag(pcov))

t0_fit, M0_fit, A1_fit, tau1_fit, A2_fit, tau2_fit = popt

# --- 4. Calculate G_sl in SI Units (W / m^3 K) ---
# Volumetric heat capacities for CrI3 at T = 15 K (in J / m^3 K)
# Update C_s and C_l with your exact DFT / experimental heat capacity values if available
C_s = 2.0e4  # Spin heat capacity at 15 K
C_l = 8.0e4  # Lattice heat capacity at 15 K

# Effective heat capacity C_eff = (C_s * C_l) / (C_s + C_l)
C_eff = (C_s * C_l) / (C_s + C_l)

# Convert tau2 from picoseconds to seconds
tau2_sec = tau2_fit * 1e-12
tau2_err_sec = perr[5] * 1e-12

# Calculate G_sl and propagate standard error
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

# Inset plot for ultrafast region
ax_inset = fig.add_axes([0.48, 0.48, 0.35, 0.35])
ax_inset.scatter(
    t_data,
    m_data,
    color="#1f77b4",
    s=20,
    alpha=0.8,
    edgecolor="k",
    linewidth=0.5,
)
t_inset = np.linspace(-2, 10, 300)
ax_inset.plot(t_inset, demag_model(t_inset, *popt), color="#d62728", linewidth=2)
ax_inset.set_xlim(-2, 10)
ax_inset.set_ylim(-0.6, 0.05)
ax_inset.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()