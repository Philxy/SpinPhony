
lin_data = """
1.1008974616569698, -0.0003616636528028794
1.0690844551604273, -0.055334538878842765
1.0523407675306817, -0.084267631103074
0.8049527828009091, -0.11175406871609406
1.4692585895117922, -0.16383363471971069
0.9920634920635011, -0.18842676311030737
1.202615364007789, -0.22459312839059686
1.408144129663128, -0.26943942133815546
1.1607561449333792, -0.2969258589511753
1.1507099323555057, -0.3142857142857143
1.140663719777649, -0.33164556962025327
1.8250619516442534, -0.3490054249547919
1.8133413703034083, -0.36925858951175417
2.0305907172995834, -0.39385171790235074
2.4818330989217134, -0.4141048824593129
3.395201259125335, -0.4358047016274864
4.314429709999345, -0.4473779385171789
5.231146607728901, -0.46329113924050624
6.613338021565897, -0.47486437613019883
7.7640479539213745, -0.4864376130198915
8.91392070189541, -0.4994575045207956
10.294437746969413, -0.5139240506329115
11.674954792043419, -0.5283905967450272
12.593346058535937, -0.5414104882459313
13.744893175272935, -0.5515370705244123
15.588373183309903, -0.566003616636528
17.428504453820928, -0.5862567811934899
19.039665795994928, -0.6021699819168174
20.649152769405948, -0.6209764918625678
22.493469961824406, -0.6339963833634719
24.107142857142875, -0.6455696202531644
26.412748643761315, -0.6614828209764919
27.328628357109384, -0.6788426763110308
28.93895251490188, -0.6962025316455696
30.550951041457385, -0.7106690777576853
32.85404527493136, -0.7309222423146474
34.7008740204943, -0.7396021699819169
36.543516844149764, -0.755515370705244
38.62098988681267, -0.7656419529837251
40.92408412028666, -0.7858951175406872
42.99820842542362, -0.8018083182640143
46.224717031679056, -0.8264014466546111
48.530322818297506, -0.8423146473779385
50.142321344852995, -0.8567811934900542
52.215608465608476, -0.8741410488245932
54.76441631504924, -0.8698010849909583
56.37725202598621, -0.8828209764918625
60.07760699216396, -0.8886075949367088
62.1542428504454, -0.9001808318264015
63.757869533185975, -0.9291139240506329
67.22925457102673, -0.9305605786618445
69.76801620788964, -0.9435804701627486
73.92965976826736, -0.9522603978300179
76.47763043332664, -0.9493670886075948
78.09046614426364, -0.962386980108499
82.2562956265488, -0.9638336347197107
84.57194762574511, -0.962386980108499
88.04333266358582, -0.9638336347197106
92.20497622396356, -0.9725135623869802
95.21088674569687, -0.9783001808318263
98.68645770544506, -0.9725135623869802
"""

log_exp = """
102.43040264272612, -0.9754954954954955
104.91987385550993, -0.9855855855855855
112.41927739901041, -0.9812612612612613
107.79292321723639, -0.9855855855855855
114.11923176004694, -0.9971171171171171
117.59664705963829, -0.9827027027027027
121.90969901941544, -0.9927927927927926
126.00215384548649, -0.9971171171171171
129.841662846783, -0.9985585585585587
136.63924023084442, -0.9884684684684685
141.226159258709, -0.9754954954954954
133.79816849393947, -0.9971171171171171
152.68944199429706, -0.9726126126126126
147.73020419788665, -0.9812612612612613
158.76542773701647, -0.9610810810810809
164.58841296973932, -0.9581981981981982
169.60371369135518, -0.9581981981981981
172.68595526863626, -0.9697297297297295
177.94800245923028, -0.9553153153153152
186.70281970862413, -0.9553153153153153
195.88836292296878, -0.9538738738738738
203.68336750703858, -0.9466666666666668
210.52092831679684, -0.9409009009009008
216.9358742091096, -0.927927927927928
225.56842444678847, -0.9221621621621623
235.24957412312185, -0.9207207207207208
245.34622813826277, -0.9135135135135135
255.10931433942753, -0.9120720720720721
262.8829473990812, -0.9005405405405404
272.5246152511977, -0.8890090090090089
278.31140583821644, -0.8774774774774774
292.0039759309533, -0.8688288288288287
304.5364666995754, -0.865945945945946
318.56162234658217, -0.8630630630630629
329.2555955026119, -0.852972972972973
339.2885971940735, -0.8443243243243241
350.67836575380574, -0.8342342342342342
362.45048382046804, -0.8241441441441442
376.8735110014648, -0.8169369369369368
"""

import io

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

plt.style.use("science")


lin_x, lin_y = np.loadtxt(io.StringIO(lin_data), delimiter=",", unpack=True)
log_x, log_y = np.loadtxt(io.StringIO(log_exp), delimiter=",", unpack=True)

idx = np.argsort(np.concatenate([lin_x, log_x]))
total_x = np.concatenate([lin_x, log_x])[idx]
total_y = np.concatenate([lin_y, log_y])[idx]

# Fit type II demagnetization model to the data
def m3tm_empirical(t, A, tau_demag, tau_remag, y0):
    return y0 + A * (1 - np.exp(-t / tau_demag)) * np.exp(-t / tau_remag)

p0 = [-1.0, 1.0, 200.0, 0.0]
popt, pcov = curve_fit(m3tm_empirical, total_x, total_y, p0=p0, maxfev=20000)
perr = np.sqrt(np.diag(pcov))
A_fit, tau_demag_fit, tau_remag_fit, y0_fit = popt
A_err, tau_demag_err, tau_remag_err, y0_err = perr

print(f"A          = {A_fit:.4f} +/- {A_err:.4f}")
print(f"tau_demag  = {tau_demag_fit:.4f} +/- {tau_demag_err:.4f} ps")
print(f"tau_remag  = {tau_remag_fit:.4f} +/- {tau_remag_err:.4f} ps")
print(f"y0         = {y0_fit:.4f} +/- {y0_err:.4f}")

t_continuous = np.linspace(total_x.min(), total_x.max(), 2000)
M_continuous = m3tm_empirical(t_continuous, *popt)

# ------------------------------------------------------------------
# Recovery-only fit: the demag onset shows two rapid components that
# a single tau_demag cannot capture, and forcing one exponential over
# the whole curve drags y0 negative to compromise -- which corrupts
# tau_remag through the shared y0/tau_remag coupling. Since only the
# long-time limit and tau_remag are of interest, fit ONLY the data
# past the minimum with a plain single-exponential relaxation toward
# y0. This is insensitive to how many components built the initial
# dip, as long as they've decayed away by the cutoff.
# ------------------------------------------------------------------
T_CUTOFF = 150.0  # ps; comfortably past the minimum, into the monotonic recovery
rec_mask = total_x > T_CUTOFF
t_rec, M_rec = total_x[rec_mask], total_y[rec_mask]

# A free y0 leaves the fit degenerate here: the recovery window (150-377 ps)
# only spans a small fraction of tau_remag, so exp(-t/tau) is nearly linear
# over that range and y0/B/tau trade off against each other freely (this
# was checked -- letting y0 float sends tau_remag to >1e6 ps with an error
# bar orders of magnitude larger than the value itself). Physically M
# recovers fully at long times (DeltaM/M0 -> 0), so fix y0 = 0 -- this
# removes the degenerate direction and leaves a well-constrained 2-param fit.
def recovery_only(t, B, tau_remag):
    return B * np.exp(-t / tau_remag)

p0_rec = [-1.0, 500.0]
popt_rec, pcov_rec = curve_fit(recovery_only, t_rec, M_rec, p0=p0_rec, maxfev=20000)
perr_rec = np.sqrt(np.diag(pcov_rec))
B_rec, tau_remag_rec = popt_rec
B_rec_err, tau_remag_rec_err = perr_rec
y0_rec, y0_rec_err = 0.0, 0.0

print()
print(f"[recovery-only fit, t > {T_CUTOFF:.0f} ps, {rec_mask.sum()} points, y0 fixed to 0]")
print(f"B                    = {B_rec:.4f} +/- {B_rec_err:.4f}")
print(f"tau_remag            = {tau_remag_rec:.2f} +/- {tau_remag_rec_err:.2f} ps")

t_rec_continuous = np.linspace(t_rec.min(), total_x.max(), 500)
M_rec_continuous = recovery_only(t_rec_continuous, *popt_rec)


# plot a recovery curve from the split with tau=160ps
tau_fixed = 160.0


def recovery_fixed_tau(t, B):
    return B * np.exp(-t / tau_fixed)


# Fit the amplitude B to the recovery region with tau fixed to 160 ps
popt_160, pcov_160 = curve_fit(recovery_fixed_tau, t_rec, M_rec, p0=[-1.0])
B_160 = popt_160[0]
B_160_err = np.sqrt(pcov_160[0, 0])

print(f"B (tau = {tau_fixed:.0f} ps fixed)   = {B_160:.4f} +/- {B_160_err:.4f}")

M_160_continuous = recovery_fixed_tau(t_rec_continuous, -2.6)

fig, ax = plt.subplots(figsize=(8 / 2.52, 6 / 2.52))

ax.plot(
    t_rec_continuous,
    M_160_continuous,
    color="blue",
    linestyle="--",
    linewidth=1.8,
    label=rf"Theory",
)


ax.scatter(total_x, total_y, label="Experiment", color="black", s=10)
ax.plot(t_continuous, M_continuous, color="red", linewidth=1.5, alpha=.9, label="Type-II fit")
ax.plot(t_rec_continuous, M_rec_continuous, color="green", linewidth=2.0, label="Recovery-only fit")
ax.axvline(T_CUTOFF, color="gray", linestyle="--", linewidth=0.8)




fit_text = (
    rf"$\tau_\mathrm{{remag}} = {tau_remag_rec:.1f} \pm {tau_remag_rec_err:.1f}$ ps" "\n"
    rf"$y_0 = 0$ (fixed)"
)

ax.legend(fontsize=10, loc="upper right")
ax.set_xlabel("Delay (ps)", fontsize=12)
#ax.set_ylabel(r"$\Delta M / M_0$", fontsize=12)
ax.set_ylabel(r"TRPR (a.u.)", fontsize=12)
fig.tight_layout()
plt.savefig("demag_fit.pdf", dpi=300)
plt.show()