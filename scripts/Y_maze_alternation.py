import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_48DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
IMG_PATH = r"C:\Users\Behaviour\Desktop\img00763.png"
OUTPUT   = r"C:\Users\Behaviour\Desktop\48_alternation_.png"

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99

# ── ZONES Y-MAZE (x, y) — conversion depuis Y X ──────────────────────────────
BRAS_1 = np.array([
    [551, 430],
    [540, 529],
    [1055, 555],
    [1061, 450],
])
BRAS_2 = np.array([
    [1061, 450],
    [1151, 504],
    [1432, 68],
    [1343, 17],
])
BRAS_3 = np.array([
    [1151, 504],
    [1055, 555],
    [1318, 1044],
    [1413, 995],
])


ZONES = {
    'Bras 1': {'poly': BRAS_1, 'color': '#00d4ff'},
    'Bras 2': {'poly': BRAS_2, 'color': '#ff6b6b'},
    'Bras 3': {'poly': BRAS_3, 'color': '#00ffaa'},
}
ARM_TO_NUM = {'Bras 1': 1, 'Bras 2': 2, 'Bras 3': 3}

# ── CHARGEMENT CSV ───────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH, header=[0, 1, 2], index_col=0)
scorer = df.columns[0][0]

def load_bodypart(name):
    x = df[(scorer, name, "x")].values.astype(float)
    y = df[(scorer, name, "y")].values.astype(float)
    p = df[(scorer, name, "likelihood")].values.astype(float)
    return x, y, p

nose_x, nose_y, nose_p = load_bodypart("nose")
tail_x, tail_y, tail_p = load_bodypart("tail_base")

total_frames_video = len(nose_x)
total_time_video    = total_frames_video / FPS

# ── INTERPOLATION À PARTIR DES POINTS FIABLES + LISSAGE ─────────────────────
# Les points avec likelihood < seuil ne sont PAS supprimés : ils sont
# reconstruits par interpolation à partir des points fiables voisins.
# Résultat : couverture de 100% de la vidéo, tout en restant guidé
# par les détections de confiance.
def interp_from_anchors(arr, reliable_mask):
    idx = np.arange(len(arr))
    if reliable_mask.sum() < 2:
        return arr
    return np.interp(idx, idx[reliable_mask], arr[reliable_mask])

def process_bodypart(x, y, p, threshold):
    reliable = p >= threshold
    x_interp = interp_from_anchors(x, reliable)
    y_interp = interp_from_anchors(y, reliable)
    x_s = gaussian_filter1d(x_interp, sigma=GAUSSIAN_SIGMA)
    y_s = gaussian_filter1d(y_interp, sigma=GAUSSIAN_SIGMA)
    pct_reliable = reliable.sum() / len(p) * 100
    return x_s, y_s, pct_reliable

nose_x_smooth, nose_y_smooth, pct_nose_reliable = process_bodypart(nose_x, nose_y, nose_p, LIKELIHOOD_THRESHOLD)
tail_x_smooth, tail_y_smooth, pct_tail_reliable = process_bodypart(tail_x, tail_y, tail_p, LIKELIHOOD_THRESHOLD)

# Couverture utilisée pour l'analyse : 100% de la vidéo (interpolée)
frame_idx = np.arange(total_frames_video)
nose_xv, nose_yv = nose_x_smooth, nose_y_smooth
tail_xv, tail_yv = tail_x_smooth, tail_y_smooth
times_v = frame_idx / FPS

frames_detected = total_frames_video
pct_detected    = 100.0

# ── ENTRÉE CONFIRMÉE = NEZ + QUEUE DANS LE MÊME BRAS ─────────────────────────
combined_in_zone = {}
for nom, zone in ZONES.items():
    path    = Path(np.vstack([zone['poly'], zone['poly'][0]]))
    in_nose = path.contains_points(np.column_stack([nose_xv, nose_yv]))
    in_tail = path.contains_points(np.column_stack([tail_xv, tail_yv]))
    combined_in_zone[nom] = in_nose & in_tail

noms = list(ZONES.keys())

# Temps + entrées par bras (corps entier confirmé)
results = {}
for nom in noms:
    in_zone   = combined_in_zone[nom]
    frames_z  = np.sum(in_zone)
    time_z    = frames_z / FPS
    pct_z     = frames_z / total_frames_video * 100
    entries_z = np.sum(np.diff(in_zone.astype(int)) == 1)
    results[nom] = {'time': time_z, 'pct': pct_z, 'entries': entries_z}

# ── SÉQUENCE DES BRAS VISITÉS (corps entier confirmé) ────────────────────────
arm_sequence = []
entry_times  = []
current_arm  = None

for i in range(len(frame_idx)):
    found = None
    for nom in noms:
        if combined_in_zone[nom][i]:
            found = nom
            break
    if found is not None:
        if found != current_arm:
            arm_sequence.append(found)
            entry_times.append(times_v[i])
            current_arm = found
    else:
        current_arm = None

# ── CLASSIFICATION DES TRIADES : ALTERNANCE CORRECTE vs ERREUR ──────────────
triads = []
for i in range(len(arm_sequence) - 2):
    triade  = arm_sequence[i:i+3]
    correct = len(set(triade)) == 3
    triads.append({
        'triade':   triade,
        'correct':  correct,
        'time_end': entry_times[i+2],
    })

total_triads   = len(triads)
correct_triads = sum(t['correct'] for t in triads)
error_triads   = total_triads - correct_triads
pct_correct    = correct_triads / total_triads * 100 if total_triads > 0 else 0

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('#1a1a2e')
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# ── AX1 : Trajectoire + zones ────────────────────────────────────────────────
try:
    img = np.array(Image.open(IMG_PATH))
    ax1.imshow(img, alpha=0.5)
except Exception:
    ax1.set_facecolor('#0f0f23')

for nom, zone in ZONES.items():
    poly_fill = patches.Polygon(zone['poly'], closed=True,
                                 facecolor=zone['color'], alpha=0.2, linewidth=0)
    poly_border = patches.Polygon(zone['poly'], closed=True,
                                   edgecolor=zone['color'], facecolor='none',
                                   linewidth=2.5, linestyle='--',
                                   label=f"{nom} ({results[nom]['time']:.1f}s | {results[nom]['entries']} entrées)")
    ax1.add_patch(poly_fill)
    ax1.add_patch(poly_border)
    cx = np.mean(zone['poly'][:, 0])
    cy = np.mean(zone['poly'][:, 1])
    ax1.text(cx, cy, nom, color=zone['color'], fontsize=9,
             ha='center', va='center', fontweight='bold')

cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
n = len(nose_xv)
for i in range(n - 1):
    t = i / n
    ax1.plot([nose_xv[i], nose_xv[i+1]], [nose_yv[i], nose_yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.7)

ax1.scatter(nose_xv[0],  nose_yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(nose_xv[-1], nose_yv[-1], color='red',  s=80, zorder=5, label='Fin')

ax1.set_title('Trajectoire Y-Maze — couverture optimale (nez)', color='white', fontsize=13, pad=10)
ax1.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', labelcolor='white')
ax1.set_xlim(0, 1920)
ax1.set_ylim(1080, 0)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

# ── AX2 : Timeline des entrées + correction des triades ─────────────────────
ax2.set_facecolor('#0f0f23')

x_seq = entry_times
y_seq = [ARM_TO_NUM[nom] for nom in arm_sequence]

ax2.plot(x_seq, y_seq, '-', color='#888888', linewidth=1, zorder=1)
for t, nom in zip(x_seq, arm_sequence):
    ax2.scatter(t, ARM_TO_NUM[nom], s=70, color=ZONES[nom]['color'],
                edgecolor='white', linewidth=1, zorder=3)

for idx, tri in enumerate(triads):
    last_time = x_seq[idx + 2]
    last_y    = y_seq[idx + 2]
    ring_color = '#00ff7f' if tri['correct'] else '#ff0033'
    ax2.scatter(last_time, last_y, s=220, facecolors='none',
                edgecolors=ring_color, linewidth=2.5, zorder=4)

legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#00ff7f', markeredgewidth=2, markersize=10,
           label='Alternance correcte', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#ff0033', markeredgewidth=2, markersize=10,
           label='Erreur (retour)', linestyle='None'),
]
ax2.legend(handles=legend_elems, fontsize=8, facecolor='#1a1a2e',
           labelcolor='white', loc='upper right')

ax2.set_yticks([1, 2, 3])
ax2.set_yticklabels(['Bras 1', 'Bras 2', 'Bras 3'], color='white')
ax2.set_xlabel('Temps (s)', color='white', fontsize=10)
ax2.set_title("Séquence des entrées dans les bras", color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# ── AX3 : Barres correct vs erreur ───────────────────────────────────────────
ax3.set_facecolor('#0f0f23')
bars = ax3.bar(['Alternance\ncorrecte', 'Erreur\n(retour)'],
               [correct_triads, error_triads],
               color=['#00ff7f', '#ff0033'], width=0.5,
               alpha=0.85, edgecolor='white', linewidth=1)
for bar, val in zip(bars, [correct_triads, error_triads]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{val}', ha='center', va='bottom',
              color='white', fontsize=13, fontweight='bold')

ax3.set_ylabel("Nombre de triades", color='white', fontsize=10)
ax3.set_title(f"Performance d'alternance — {pct_correct:.1f}% correct",
              color='white', fontsize=13, pad=10)
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')

# ── AX4 : Stats texte ────────────────────────────────────────────────────────
ax4.set_facecolor('#0f0f23')
ax4.axis('off')

stats_text = (
    f"STATISTIQUES Y-MAZE — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD})\n"
    f"   Nez   : {pct_nose_reliable:.1f}% des frames\n"
    f"   Queue : {pct_tail_reliable:.1f}% des frames\n\n"
    f"Bras 1 : {results['Bras 1']['time']:.1f}s | {results['Bras 1']['entries']} entrées\n"
    f"Bras 2 : {results['Bras 2']['time']:.1f}s | {results['Bras 2']['entries']} entrées\n"
    f"Bras 3 : {results['Bras 3']['time']:.1f}s | {results['Bras 3']['entries']} entrées\n\n"
    f"Total entrées (séquence) : {len(arm_sequence)}\n"
    f"Total triades évaluées   : {total_triads}\n\n"
    f"Alternances correctes : {correct_triads}\n"
    f"Erreurs (retour)      : {error_triads}\n"
    f"% Alternance correcte : {pct_correct:.1f}%\n\n"
    f"Lissage gaussien : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation à partir des points fiables\n"
    f"          (pas de frames supprimées)"
)
ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
          fontsize=10, color='white', verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Analyse Y-Maze — Alternance spontanée Video_48',
             color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')

print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé Y-Maze (couverture optimale) :")
print(f"   Durée totale vidéo     : {total_time_video:.1f} s")
print(f"   Fiabilité nez brute    : {pct_nose_reliable:.1f}%")
print(f"   Fiabilité queue brute  : {pct_tail_reliable:.1f}%")
for nom in noms:
    print(f"   {nom} → Temps : {results[nom]['time']:.1f}s | Entrées : {results[nom]['entries']}")
print(f"   Séquence des bras      : {' -> '.join(arm_sequence)}")
print(f"   Total triades          : {total_triads}")
print(f"   Alternances correctes  : {correct_triads}")
print(f"   Erreurs (retour)       : {error_triads}")
print(f"   % Alternance correcte  : {pct_correct:.1f}%")
