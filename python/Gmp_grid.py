import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

plt.style.use("science")

# -------------------------------------------------------------------------
# Raw Data Strings
# -------------------------------------------------------------------------
bccFe_data = """
T_K,G_mp_vol_W_m3_K
100.00,1.964040e+15
108.08,2.264311e+15
116.16,2.581427e+15
124.24,2.915029e+15
132.32,3.264718e+15
140.40,3.630043e+15
148.48,4.010579e+15
156.57,4.405856e+15
164.65,4.815407e+15
172.73,5.238778e+15
180.81,5.675514e+15
188.89,6.125174e+15
196.97,6.587329e+15
205.05,7.061559e+15
213.13,7.547460e+15
221.21,8.044675e+15
229.29,8.552834e+15
237.37,9.071590e+15
245.45,9.600609e+15
253.54,1.013961e+16
261.62,1.068827e+16
269.70,1.124634e+16
277.78,1.181360e+16
285.86,1.238980e+16
293.94,1.297474e+16
302.02,1.356822e+16
310.10,1.416993e+16
318.18,1.477997e+16
326.26,1.539825e+16
334.34,1.602486e+16
342.42,1.665913e+16
350.51,1.730132e+16
358.59,1.795136e+16
366.67,1.860924e+16
374.75,1.927500e+16
382.83,1.994840e+16
390.91,2.062999e+16
398.99,2.131948e+16
407.07,2.201701e+16
415.15,2.272261e+16
423.23,2.343644e+16
431.31,2.415860e+16
439.39,2.488921e+16
447.47,2.562838e+16
455.56,2.637624e+16
463.64,2.713300e+16
471.72,2.789876e+16
479.80,2.867368e+16
487.88,2.945793e+16
495.96,3.025168e+16
504.04,3.105508e+16
512.12,3.186830e+16
520.20,3.269150e+16
528.28,3.352484e+16
536.36,3.436849e+16
544.44,3.522260e+16
552.53,3.608731e+16
560.61,3.696276e+16
568.69,3.784908e+16
576.77,3.874641e+16
584.85,3.965486e+16
592.93,4.057454e+16
601.01,4.150555e+16
609.09,4.244800e+16
617.17,4.340209e+16
625.25,4.436775e+16
633.33,4.534510e+16
641.41,4.633419e+16
649.49,4.733510e+16
657.58,4.834785e+16
665.66,4.937248e+16
673.74,5.040901e+16
681.82,5.145746e+16
689.90,5.251783e+16
697.98,5.359005e+16
706.06,5.467423e+16
714.14,5.577033e+16
722.22,5.687825e+16
730.30,5.799797e+16
738.38,5.912947e+16
746.46,6.027271e+16
754.55,6.142759e+16
762.63,6.259407e+16
770.71,6.377207e+16
778.79,6.496150e+16
786.87,6.616227e+16
794.95,6.737430e+16
803.03,6.859747e+16
811.11,6.983178e+16
819.19,7.107700e+16
827.27,7.233303e+16
835.35,7.359980e+16
843.43,7.487717e+16
851.52,7.616503e+16
859.60,7.746332e+16
867.68,7.877173e+16
875.76,8.008988e+16
883.84,8.141786e+16
891.92,8.275551e+16
900.00,8.410314e+16
"""

exp_data = """
78.06035010765932, 0.044774422771366854
88.77461661638372, 0.10882891049031208
91.03072214342009, 0.18573217243065632
103.2009988696262, 0.2711270525334949
118.8479794695147, 0.43325569629717886
138.59555307496296, 0.6980509757919167
161.62434917189353, 1.3700748095412656
189.66582214551497, 1.827024122190361
221.8753057729327, 2.517850823588334
263.65793664694536, 3.6155373539268707
261.1882620278661, 4.226978079673294
313.3088980531566, 4.55169185214958
260.37018864314103, 5.27786852038749
261.1882620278658, 6.324555320336756
309.4020150772508, 7.03815078807685
306.503857284445, 14.875054551318087
189.66582214551497, 2.262566873264038
161.11812206446368, 1.8120611644906752
118.8479794695147, 0.6020068379164274
138.59555307496296, 1.0705363645694208
221.8753057729327, 3.829771410664671
223.2717439180504, 3.0672154375400233
189.0717657362239, 2.4767783257853426
89.05354313816214, 0.12828410624473863
118.47573315156203, 0.4703905270141916
118.47573315156203, 0.582526530981748
137.29733340405326, 0.8296351395932767
137.29733340405326, 0.7516748085768887
160.61348052186742, 1.3927948059248652
"""

# -------------------------------------------------------------------------
# 1. Physical Parameters & Constants
# -------------------------------------------------------------------------
file_path_bccFe = (
    "Outputs/bccFe_Full_NonHyrbid_sig_30.0_300K/Gmp_temperature_grid.csv"
)
file_path_CrI3 = "Outputs/CrI3_Full_Hyrbid_sig_1.0_15K/Gmp_temperature_grid.csv"

# Energy conversion: 1 meV/(ps*K) -> W/K
EV_TO_JOULE = 1.602176634e-19
CONV_W_PER_K = (1e-3 * EV_TO_JOULE) / 1e-12  # 1.602176634e-10 W/K

# Volumes
a_bccFe = 2.8665 * 1e-10  # meters
V_bccFe = a_bccFe**3  # ~ 2.355e-29 m^3
T_C_bccFe = 1043.0  # Curie temperature (K)

V_CrI3 = 2.8179e-28  # 281.79 Å^3
T_C_CrI3 = 61.0  # Curie temperature (K)

# --- Third Material Parameters (Exp / Non-FM) ---
# Characteristic scale: Set to Debye temperature (Theta_D), Neel Temp (T_N), or 300 K
T_char_exp = 300.0  # e.g., Theta_D or T_ref = 300 K
EXP_LABEL = r"$\mathrm{Exp\ Material}\ (T_{\mathrm{char}} = 300\ \mathrm{K})$"

# Scale factor for exp data (set to 1e16 if exp values are in 10^16 W/m^3/K)
EXP_UNIT_FACTOR = 1e16

# -------------------------------------------------------------------------
# 2. Data Loading and Processing
# -------------------------------------------------------------------------
def process_sim_data(file_path, V_cell, T_char):
    vol_factor = CONV_W_PER_K / V_cell
    df = pd.read_csv(file_path)
    df["G_mp_vol"] = df["G_mp"] * vol_factor

    df_eq = df[np.isclose(df["T_m_K"], df["T_p_K"])].copy()
    df_eq = df_eq.sort_values(by="T_m_K")
    df_eq["T_K"] = df_eq["T_m_K"]
    df_eq["T_norm"] = df_eq["T_K"] / T_char
    return df_eq


df_eq_bccFe = process_sim_data(file_path_bccFe, V_bccFe, T_C_bccFe)
df_eq_CrI3 = process_sim_data(file_path_CrI3, V_CrI3, T_C_CrI3)

# Load second bcc-Fe dataset
df_bccFe_extra = pd.read_csv(io.StringIO(bccFe_data.strip()))
df_bccFe_extra["T_norm"] = df_bccFe_extra["T_K"] / T_C_bccFe

# Load and sort experimental dataset
df_exp = pd.read_csv(
    io.StringIO(exp_data.strip()), header=None, names=["T_K", "G_raw"]
)
df_exp = df_exp.sort_values(by="T_K")
df_exp["G_mp_vol"] = df_exp["G_raw"] * EXP_UNIT_FACTOR
df_exp["T_norm"] = df_exp["T_K"] / T_char_exp

# -------------------------------------------------------------------------
# 3. Plot 1: Characteristic Temperature Normalization (T / T_char)
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 5))

# bcc-Fe (sig 30.0)
plt.plot(
    df_eq_bccFe["T_norm"],
    df_eq_bccFe["G_mp_vol"],
    marker="s",
    linestyle="-",
    linewidth=1.5,
    markersize=4,
    color="crimson",
    label=r"$\mathrm{bcc\text{-}Fe}\ (\sigma = 30.0,\ T_C = 1043\ \mathrm{K})$",
)

# bcc-Fe (sig 4.0)
plt.plot(
    df_bccFe_extra["T_norm"],
    df_bccFe_extra["G_mp_vol_W_m3_K"],
    marker="^",
    linestyle="--",
    linewidth=1.5,
    markersize=4,
    color="darkorange",
    label=r"$\mathrm{bcc\text{-}Fe}\ (\sigma = 4.0,\ T_C = 1043\ \mathrm{K})$",
)

# CrI3
plt.plot(
    df_eq_CrI3["T_norm"],
    df_eq_CrI3["G_mp_vol"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=4,
    color="navy",
    label=r"$\mathrm{CrI}_3\ (T_C = 61\ \mathrm{K})$",
)

# Experimental / Non-Curie Material
plt.scatter(
    df_exp["T_norm"],
    df_exp["G_mp_vol"],
    marker="d",
    color="forestgreen",
    s=25,
    alpha=0.85,
    label=EXP_LABEL,
    zorder=5,
)

plt.yscale("log")
plt.xlabel(r"$T / T_{\mathrm{char}}$", fontsize=12)
plt.ylabel(r"$G_{mp}$ ($\mathrm{W\cdot m^{-3}\cdot K^{-1}}$)", fontsize=12)
plt.title(
    r"$\mathrm{Magnon\text{-}Phonon\ Coupling\ vs.\ Normalized\ Temperature}$",
    fontsize=13,
    pad=10,
)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 4. Plot 2: Absolute Temperature Comparison (T in Kelvin)
# -------------------------------------------------------------------------
plt.figure(figsize=(7, 5))

# bcc-Fe (sig 30.0)
plt.plot(
    df_eq_bccFe["T_K"],
    df_eq_bccFe["G_mp_vol"],
    marker="s",
    linestyle="-",
    linewidth=1.5,
    markersize=4,
    color="crimson",
    label=r"$\mathrm{bcc\text{-}Fe}\ (\sigma = 30.0)$",
)

# bcc-Fe (sig 4.0)
plt.plot(
    df_bccFe_extra["T_K"],
    df_bccFe_extra["G_mp_vol_W_m3_K"],
    marker="^",
    linestyle="--",
    linewidth=1.5,
    markersize=4,
    color="darkorange",
    label=r"$\mathrm{bcc\text{-}Fe}\ (\sigma = 4.0)$",
)

# CrI3
plt.plot(
    df_eq_CrI3["T_K"],
    df_eq_CrI3["G_mp_vol"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    markersize=4,
    color="navy",
    label=r"$\mathrm{CrI}_3$",
)

# Experimental / Non-Curie Material
plt.scatter(
    df_exp["T_K"],
    df_exp["G_mp_vol"],
    marker="d",
    color="forestgreen",
    s=25,
    alpha=0.85,
    label=r"$\mathrm{Exp\ Material}$",
    zorder=5,
)

plt.yscale("log")
plt.xlabel(r"$T\ (\mathrm{K})$", fontsize=12)
plt.ylabel(r"$G_{mp}$ ($\mathrm{W\cdot m^{-3}\cdot K^{-1}}$)", fontsize=12)
plt.title(
    r"$\mathrm{Magnon\text{-}Phonon\ Coupling\ vs.\ Absolute\ Temperature}$",
    fontsize=13,
    pad=10,
)
plt.legend(frameon=True, fontsize=9)
plt.tight_layout()
plt.show()