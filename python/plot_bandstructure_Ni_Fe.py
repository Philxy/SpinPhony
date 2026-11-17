import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import scienceplots
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from scipy.interpolate import CubicSpline

plt.style.use(['science', 'no-latex'])

# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------
MAG_LEFT = True            # True: magnon energy on the left axis, phonon energy on the right; False: the other way round
PHONON_COLOR = 'cornflowerblue'
MAGNON_COLOR = 'crimson'

NI_DATA_FILE = 'band.csv'
NI_YAML_FILE = 'Inputs/meshes/bandNi.yaml'

FE_PHONON_YAML_FILE = 'Outputs/bandFe_acc.yaml'
FE_MAGNON_DAT_FILE = 'Outputs/Lifetimes/temp_300_stddev_3.002_scaling_1.0_mesh_mesh48Fe/lifetimes_magnons.dat'
FE_PHONON_EXP_FILES = {
    'GH': 'Inputs/bccFe_phonon_disp_exp_GH.txt',
    'HP': 'Inputs/bccFe_phonon_disp_exp_HP.txt',
    'PG': 'Inputs/bccFe_phonon_disp_exp_PG.txt',
    'GN': 'Inputs/bccFe_phonon_disp_exp_GN.txt',
}

# common axis limits of both panels (meV); None: data maximum x 1.05
PHONON_LIM = None
MAGNON_LIM = None

OUTPUT_STEM = 'bandstructure_Ni_Fe'

# ----------------------------------------------------------------------------------------------------------------------
# Experimental data: fcc Ni (see PHYSICAL REVIEW B 107, 104303 (2023) and the references therein)
# ----------------------------------------------------------------------------------------------------------------------
GammaX = """0.20625000000000013, 60.79280821917811
0.17500000000000007, 131.4948630136987
0.2625, 106.24143835616445
0.2843750000000001, 123.21575342465762
0.31874999999999987, 131.96404109589045
0.284375, 147.87328767123293
0.409375, 147.80479452054797
0.34687500000000004, 171.9486301369864
0.40312499999999996, 185.06849315068501
0.46250000000000024, 205.85787671232885
0.6281250000000002, 246.31506849315068
0.5125000000000001, 172.40582191780825
"""

GammaL = """-0.14999999999999997, 39.61815068493154
-0.19687500000000005, 60.46575342465759
-0.21562500000000007, 81.84589041095894
-0.259375, 103.23972602739728
-0.3, 123.53595890410965
-0.409375, 150.44520547945206
-0.4999999999999999, 164.74143835616442
-0.55, 175.72773972602738
"""

GammaL_phonon = """
0.19993510896286626, 5.4315295566195125
0.3032657662341989, 7.77731658922726
0.4033518716488568, 10.20810688718527
0.5034629351547192, 12.383474083102131
0.6036322342067281, 13.962854040922954
0.7038056929406038, 15.499663481736919
0.7458060007570626, 16.225153263478415
0.7975441238254098, 16.653062732162248
0.8395777090968091, 17.037988377848865
0.8977633390598289, 17.721596485900765
0.9366106080206996, 17.595538158841617
0.9413110485310094, 35.73085394109059
0.9056251377894614, 36.623319342936654
0.8409753622042994, 35.85429466354413
0.7989459366147679, 35.426798500850666
0.695540405069819, 33.847280774366375
0.6019808405053185, 30.8633492926461
0.5019820884098776, 27.538578137544036
0.4085431546195344, 23.32010166262484
0.29899793263811236, 18.334667038264914
0.2057045877131321, 12.626222468105643
"""

W_X_phonon = """
0.18784530386740367, 26.308688946015426
0.3756906077348073, 26.52154241645245
0.37569060773480745, 26.94724935732649
0.1878453038674039, 26.734395886889477
0.7790055248618787, 25.75526992287918
0.5856353591160227, 26.649254498714658
0.9613259668508288, 25.797840616966575
0.7790055248618786, 29.50149100257069
0.5635359116022103, 28.181799485861188
0.9613259668508297, 31.20431876606684
0.7734806629834263, 32.60915167095116
0.5911602209944752, 33.92884318766067
0.38121546961325947, 34.82282776349616
0.1823204419889501, 35.418817480719795
"""

# ----------------------------------------------------------------------------------------------------------------------
# Experimental data: bcc Fe magnons
# ----------------------------------------------------------------------------------------------------------------------
Gamma_H_magnons = """
0.1474602417198847, 29.551530522201237
0.3139253662075862, 103.31596802342685
0.29482243589324586, 94.9884630720257
0.2757227738304562, 85.46477805303724
0.26756848621124674, 69.93404668370516
0.1883983606450222, 46.20000392190193
0.22116258244164538, 54.49482635779509
0.17203095687868913, 36.66978239981176
"""

Gamma_N_magnons = """
0.24405029916771498, 35.03209844451349
0.28664862531390545, 44.66395272732334
0.44148242360663026, 97.54399320432056
0.3834220701756013, 77.11517840979243
0.36019885717468675, 68.70413264571947
0.329228384030153, 59.08620393536677
"""


def parse_xy(text):
    return np.array([list(map(float, line.split(','))) for line in text.strip().split('\n')])


# ----------------------------------------------------------------------------------------------------------------------
# fcc Ni: body of the original plot_bandstructure_dual_axis; ax_ph / ax_mag are the phonon / magnon axes of this panel
# ----------------------------------------------------------------------------------------------------------------------
def load_band_data(data_filepath, yaml_filepath):
    with open(yaml_filepath, 'r') as f:
        path_info = yaml.safe_load(f)
    segment_nqpoint = path_info.get('segment_nqpoint', [])
    labels = path_info.get('labels', [])
    if not segment_nqpoint or not labels:
        sys.exit(f"Error: YAML file '{yaml_filepath}' is missing 'segment_nqpoint' or 'labels'.")
    with open(data_filepath, 'r') as f:
        header_line = f.readline().strip()
    column_names = header_line.lstrip('#').strip().split(',')
    df = pd.read_csv(data_filepath, comment='#', names=column_names, header=None, sep=',')
    if 'Distance' not in df.columns:
        sys.exit("Error: 'Distance' column not found in data file.")
    x_data = pd.Series(np.arange(len(df)))                          # equidistant x-axis layout
    energy_columns = [col for col in df.columns if col.endswith('_Energy(meV)')]
    if not energy_columns:
        sys.exit("Error: No columns ending in '_Energy(meV)' found to plot.")
    segment_end_indices = np.cumsum(segment_nqpoint) - 1
    symmetry_point_indices = np.insert(segment_end_indices, 0, 0)
    tick_positions = x_data.iloc[symmetry_point_indices].values
    tick_labels = [labels[0][0]]
    tick_labels.extend([pair[1] for pair in labels])
    return df, x_data, energy_columns, tick_positions, tick_labels


def draw_ni(ax_x, ax_ph, ax_mag, data_filepath, yaml_filepath):
    df, x_data, energy_columns, tick_positions, tick_labels = load_band_data(data_filepath, yaml_filepath)
    phonon_max, magnon_max = 0.0, 0.0
    for col in energy_columns:
        if 'Phonon' in col:
            ax_ph.plot(x_data, df[col], color=PHONON_COLOR, linewidth=1.5)
            phonon_max = max(phonon_max, df[col].max())
        elif 'Magnon' in col:
            ax_mag.plot(x_data, df[col], color=MAGNON_COLOR, linewidth=2.0, linestyle='solid')
            magnon_max = max(magnon_max, df[col].max())
        else:
            ax_ph.plot(x_data, df[col], color='gray', linewidth=1.0, linestyle=':')
            phonon_max = max(phonon_max, df[col].max())

    ax_x.set_xlim(x_data.min(), x_data.max())
    ax_x.set_xticks(tick_positions)
    ax_x.set_xticklabels(tick_labels)
    for pos in tick_positions[1:-1]:
        ax_x.axvline(x=pos, color='black', linestyle='solid', linewidth=0.6, zorder=0)
    ax_x.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=0)

    # MAPPING EXPERIMENTAL DATA TO THE NEW X-AXIS
    orig_dist = df['Distance'].values
    new_dist = x_data.values

    gamma_data = parse_xy(GammaX)
    gamma_x_mapped = np.interp(gamma_data[:, 0] / 4, orig_dist, new_dist)
    ax_mag.scatter(gamma_x_mapped, gamma_data[:, 1], color=MAGNON_COLOR, marker='o', s=20)

    gamma_l_data = parse_xy(GammaL)
    gamma_l_mapped = np.interp(0.7444317 - gamma_l_data[:, 0] / 4, orig_dist, new_dist)
    ax_mag.scatter(gamma_l_mapped, gamma_l_data[:, 1], color=MAGNON_COLOR, marker='o', s=20)

    gamma_l_phonon_data = parse_xy(GammaL_phonon)
    gamma_l_phonon_mapped = np.interp(0.7444317 + gamma_l_phonon_data[:, 0] / 4, orig_dist, new_dist)
    ax_ph.scatter(gamma_l_phonon_mapped, gamma_l_phonon_data[:, 1], color=PHONON_COLOR, marker='x', s=30)

    w_x_phonon_data = parse_xy(W_X_phonon)
    w_x_phonon_mapped = np.interp(0.25 + w_x_phonon_data[:, 0] / 5.5, orig_dist, new_dist)
    ax_ph.scatter(w_x_phonon_mapped, w_x_phonon_data[:, 1], color=PHONON_COLOR, marker='x', s=30)

    ax_x.set_title('fcc Ni')
    return (max(phonon_max, gamma_l_phonon_data[:, 1].max(), w_x_phonon_data[:, 1].max()),
            max(magnon_max, gamma_data[:, 1].max(), gamma_l_data[:, 1].max()))


# ----------------------------------------------------------------------------------------------------------------------
# bcc Fe: body of the original plot_iron_combined_dual_axis
# ----------------------------------------------------------------------------------------------------------------------
def retrieve_exp_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',')
        return data[:, 0], data[:, 1]
    except (IOError, ValueError) as e:
        print(f"Error: Could not process file '{file_path}'. Reason: {e}")
        return None, None


def draw_fe(ax_x, ax_ph, ax_mag, phonon_yaml_file, magnon_dat_file, phonon_exp_files):
    with open(phonon_yaml_file, 'r') as f:
        phonon_data = yaml.safe_load(f)

    num_bands_ph = len(phonon_data['phonon'][0]['band'])
    ph_frequencies = [[] for _ in range(num_bands_ph)]
    for point in phonon_data['phonon']:
        for i, band in enumerate(point['band']):
            ph_frequencies[i].append(band['frequency'] * 4.136)         # THz -> meV (1 THz = 4.136 meV)

    total_q_points = len(ph_frequencies[0])
    x_axis_ph = np.arange(total_q_points)

    labels = phonon_data['labels']
    segment_nqpoint = phonon_data['segment_nqpoint']
    q_points_per_segment = segment_nqpoint[0]

    tick_labels = [labels[0][0]]
    tick_labels.extend([pair[1] for pair in labels])
    tick_labels = [r'$\Gamma$' if label == 'G' else label for label in tick_labels]

    cumulative_segments = np.cumsum(segment_nqpoint)
    tick_positions = np.insert(cumulative_segments, 0, 0)
    tick_positions[-1] = total_q_points - 1

    exp_data = {}
    for key, path in phonon_exp_files.items():
        x, y = retrieve_exp_data(path)
        if x is not None:
            exp_data[key] = (x, y)

    magnon_raw_data = np.loadtxt(magnon_dat_file, delimiter=',', skiprows=1)
    magnon_raw_data = magnon_raw_data[magnon_raw_data[:, 0].argsort()]
    x_mag_raw = magnon_raw_data[:, 0]
    y_mag_raw = magnon_raw_data[:, 2]

    spl_y = CubicSpline(x_mag_raw, y_mag_raw)
    x_smooth_orig = np.linspace(x_mag_raw.min(), x_mag_raw.max(), 1000)
    y_smooth_mag = spl_y(x_smooth_orig)
    x_smooth_mag = x_smooth_orig / x_mag_raw.max() * (total_q_points - 1)       # rescale magnon x to the phonon x range

    ax_mag.plot(x_smooth_mag, y_smooth_mag, color=MAGNON_COLOR, linewidth=2.0, linestyle='solid')
    magnon_max = np.max(y_smooth_mag)

    phonon_max = 0.0
    for i in range(num_bands_ph):
        ax_ph.plot(x_axis_ph, ph_frequencies[i], color=PHONON_COLOR, linewidth=1.5)
        phonon_max = max(phonon_max, np.max(ph_frequencies[i]))

    if 'GH' in exp_data:
        ax_ph.scatter(np.array(exp_data['GH'][0]) * q_points_per_segment, exp_data['GH'][1], color=PHONON_COLOR, marker='x', s=20)
    if 'HP' in exp_data:
        ax_ph.scatter((np.array(exp_data['HP'][0]) + 1) * q_points_per_segment, exp_data['HP'][1], color=PHONON_COLOR, marker='x', s=20)
    if 'PG' in exp_data:
        ax_ph.scatter((np.array(exp_data['PG'][0]) + 2) * q_points_per_segment, exp_data['PG'][1], color=PHONON_COLOR, marker='x', s=20)
    if 'GN' in exp_data:
        ax_ph.scatter((np.array(exp_data['GN'][0]) + 3) * q_points_per_segment, exp_data['GN'][1], color=PHONON_COLOR, marker='x', s=20)

    gamma_h_mag_data = parse_xy(Gamma_H_magnons)
    gamma_n_mag_data = parse_xy(Gamma_N_magnons)
    ax_mag.scatter(gamma_h_mag_data[:, 0] * q_points_per_segment, gamma_h_mag_data[:, 1], color=MAGNON_COLOR, marker='o', s=20)
    ax_mag.scatter((gamma_n_mag_data[:, 0] + 3) * q_points_per_segment, gamma_n_mag_data[:, 1], color=MAGNON_COLOR, marker='o', s=20)

    ax_x.set_xlim(0, total_q_points - 1)
    ax_x.set_xticks(tick_positions)
    ax_x.set_xticklabels(tick_labels)
    for pos in tick_positions[1:-1]:
        ax_x.axvline(x=pos, color='black', linestyle='solid', linewidth=0.6, zorder=0)
    ax_x.axhline(y=0, color='black', linestyle='-', linewidth=0.5, zorder=0)

    ax_x.set_title('bcc Fe')
    return phonon_max, max(magnon_max, gamma_h_mag_data[:, 1].max(), gamma_n_mag_data[:, 1].max())


# ----------------------------------------------------------------------------------------------------------------------
# One figure: Ni (left panel) and Fe (right panel); the y axes are shared between the panels, and only the OUTER axes carry
# tick labels and a label (nothing between the panels).
# ----------------------------------------------------------------------------------------------------------------------
def plot_ni_fe_side_by_side():
    # primary axes: left of each panel, shared between the panels; twin axes: right of each panel, shared as well
    fig, (ax1_ni, ax1_fe) = plt.subplots(1, 2, sharey=True, figsize=(18 / 2.52, 8 / 2.52))
    ax2_ni, ax2_fe = ax1_ni.twinx(), ax1_fe.twinx()
    ax2_fe.sharey(ax2_ni)

    if MAG_LEFT:
        ni_mag, ni_ph, fe_mag, fe_ph = ax1_ni, ax2_ni, ax1_fe, ax2_fe
    else:
        ni_ph, ni_mag, fe_ph, fe_mag = ax1_ni, ax2_ni, ax1_fe, ax2_fe
    left_color, right_color = (MAGNON_COLOR, PHONON_COLOR) if MAG_LEFT else (PHONON_COLOR, MAGNON_COLOR)
    left_label, right_label = (('Magnon energy (meV)', 'Phonon energy (meV)') if MAG_LEFT
                               else ('Phonon energy (meV)', 'Magnon energy (meV)'))

    ph_ni, mag_ni = draw_ni(ax1_ni, ni_ph, ni_mag, NI_DATA_FILE, NI_YAML_FILE)
    ph_fe, mag_fe = draw_fe(ax1_fe, fe_ph, fe_mag, FE_PHONON_YAML_FILE, FE_MAGNON_DAT_FILE, FE_PHONON_EXP_FILES)

    ph_lim = PHONON_LIM if PHONON_LIM else 1.05 * max(ph_ni, ph_fe)
    mag_lim = MAGNON_LIM if MAGNON_LIM else 1.05 * max(mag_ni, mag_fe)
    ni_ph.set_ylim(0, ph_lim)          # shared with the Fe panel
    ni_mag.set_ylim(0, mag_lim)

    # labels only on the outer axes: left of the Ni panel, right of the Fe panel; nothing in the middle
    ax1_ni.set_ylabel(left_label, color=left_color)
    ax1_ni.tick_params(axis='y', labelcolor=left_color)
    ax2_fe.set_ylabel(right_label, color=right_color)
    ax2_fe.tick_params(axis='y', labelcolor=right_color)
    ax2_ni.tick_params(axis='y', labelright=False, right=False)       # Ni panel, right side: no ticks/labels
    ax1_fe.tick_params(axis='y', labelleft=False, left=False)         # Fe panel, left side: no ticks/labels

    # legend (Ni panel)
    handle_cross = Line2D([0], [0], marker='x', color=PHONON_COLOR, linestyle='None', markersize=7)
    handle_circle = Line2D([0], [0], marker='o', color=MAGNON_COLOR, linestyle='None', markersize=7)
    handle_mag = Line2D([0], [0], color=MAGNON_COLOR, lw=3)
    handle_phon = Line2D([0], [0], color=PHONON_COLOR, lw=3)
    custom_handler = HandlerTuple(ndivide=None, pad=0.75)
    ax1_ni.legend([(handle_cross, handle_circle), (handle_phon, handle_mag)], ['neutron scattering', 'ab-initio'],
                  handler_map={tuple: custom_handler}, frameon=True, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)                                  # panels touch: no gap, no labels in between
    plt.savefig(OUTPUT_STEM + '.png', dpi=300)
    plt.savefig(OUTPUT_STEM + '.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_ni_fe_side_by_side()
