import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_47DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
IMG_PATH = r"C:\Users\Behaviour\Desktop\img00763.png"
OUTPUT   = r"C:\Users\Behaviour\Desktop\47.png"

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

# ── CHARGEMENT CSV ───────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH, header=[0,1,2], index_col=0)
scorer = df.columns[0][0]

nose_x = df[(scorer, "nose", "x")].values.astype(float)
nose_y = df[(scorer, "nose", "y")].values.astype(float)
nose_p = df[(scorer, "nose", "likelihood")].values.astype(float)

total_frames_video = len(nose_x)
total_time_video   = total_frames_video / FPS

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

reliable     = nose_p >= LIKELIHOOD_THRESHOLD
pct_reliable = reliable.sum() / total_frames_video * 100

nose_x_interp = interp_from_anchors(nose_x, reliable)
nose_y_interp = interp_from_anchors(nose_y, reliable)
nose_x_smooth = gaussian_filter1d(nose_x_interp, sigma=GAUSSIAN_SIGMA)
nose_y_smooth = gaussian_filter1d(nose_y_interp, sigma=GAUSSIAN_SIGMA)

xv = nose_x_smooth
yv = nose_y_smooth

frames_detected = total_frames_video
pct_detected    = 100.0

# ── TEMPS + ENTRÉES PAR ZONE ─────────────────────────────────────────────────
def analyze_zone(poly, x, y):
    path    = Path(np.vstack([poly, poly[0]]))
    points  = np.column_stack([x, y])
    in_zone = path.contains_points(points)
    frames_z = np.sum(in_zone)
    time_z   = frames_z / FPS
    pct_z    = frames_z / total_frames_video * 100
    entries  = np.sum(np.diff(in_zone.astype(int)) == 1)
    return time_z, pct_z, entries, in_zone

results = {}
in_zones_all = {}
for nom, zone in ZONES.items():
    time_z, pct_z, entries, in_zone = analyze_zone(zone['poly'], xv, yv)
    results[nom]      = {'time': time_z, 'pct': pct_z, 'entries': entries}
    in_zones_all[nom] = in_zone

# ── ALTERNANCE SPONTANÉE ─────────────────────────────────────────────────────
arm_sequence = []
current_arm  = None
noms         = list(ZONES.keys())

for i in range(len(xv)):
    for nom in noms:
        if in_zones_all[nom][i]:
            if nom != current_arm:
                arm_sequence.append(nom)
                current_arm = nom
            break
    else:
        current_arm = None

alternations = 0
if len(arm_sequence) >= 3:
    for i in range(len(arm_sequence) - 2):
        triade = arm_sequence[i:i+3]
        if len(set(triade)) == 3:
            alternations += 1
    max_alternations = len(arm_sequence) - 2
    pct_alternance   = alternations / max_alternations * 100 if max_alternations > 0 else 0
else:
    pct_alternance = 0

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')

ax1 = axes[0]
try:
    img = np.array(Image.open(IMG_PATH))
    ax1.imshow(img, alpha=0.5)
except Exception:
    ax1.set_facecolor('#0f0f23')

for nom, zone in ZONES.items():
    poly_fill   = patches.Polygon(zone['poly'], closed=True,
                                  facecolor=zone['color'], alpha=0.2, linewidth=0)
    poly_border = patches.Polygon(zone['poly'], closed=True,
                                  edgecolor=zone['color'], facecolor='none',
                                  linewidth=2.5, linestyle='--',
                                  label=f"{nom} ({results[nom]['time']:.1f}s | {results[nom]['entries']} entrées)")
    ax1.add_patch(poly_fill)
    ax1.add_patch(poly_border)
    cx = np.mean(zone['poly'][:, 0])
    cy = np.mean(zone['poly'][:, 1])
    ax1.text(cx, cy, nom, color=zone['color'],
             fontsize=9, ha='center', va='center', fontweight='bold')

n    = len(xv)
cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
for i in range(n - 1):
    t = i / n
    ax1.plot([xv[i], xv[i+1]], [yv[i], yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.75)

ax1.scatter(xv[0],  yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(xv[-1], yv[-1], color='red',  s=80, zorder=5, label='Fin')

ax1.set_title('Trajectoire Y-Maze — couverture optimale (nose)', color='white', fontsize=13, pad=10)
ax1.legend(loc='upper right', fontsize=7, facecolor='#1a1a2e', labelcolor='white')
ax1.set_xlim(0, 1920)
ax1.set_ylim(1080, 0)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, total_time_video))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_label('Temps (s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

ax2 = axes[1]
ax2.set_facecolor('#0f0f23')

temps   = [results[n]['time']    for n in noms]
entrees = [results[n]['entries'] for n in noms]
colors  = [ZONES[n]['color']     for n in noms]
x_pos   = np.arange(len(noms))
width   = 0.35

bars1 = ax2.bar(x_pos - width/2, temps, width,
                color=colors, alpha=0.85, edgecolor='white', linewidth=1)
for bar, val in zip(bars1, temps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}s', ha='center', va='bottom',
             color='white', fontsize=9, fontweight='bold')

ax2b = ax2.twinx()
ax2b.set_facecolor('#0f0f23')
bars2 = ax2b.bar(x_pos + width/2, entrees, width,
                 color=colors, alpha=0.45, edgecolor='white', linewidth=1)
for bar, val in zip(bars2, entrees):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{val}', ha='center', va='bottom',
              color='white', fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(noms, color='white', fontsize=10)
ax2.set_ylabel('Temps (s)', color='white', fontsize=10)
ax2b.set_ylabel('Nombre d\'entrées', color='white', fontsize=10)
ax2.set_title('Temps & entrées par bras', color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
ax2b.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

ax2.legend(['Temps (barres pleines)', 'Entrées (barres transparentes)'],
           fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

stats_text = (
    f"STATISTIQUES Y-MAZE — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD}) : {pct_reliable:.1f}%\n\n"
    f"Bras 1 (bleu)\n"
    f"   Temps    : {results['Bras 1']['time']:.1f} s ({results['Bras 1']['pct']:.1f}%)\n"
    f"   Entrées  : {results['Bras 1']['entries']}\n\n"
    f"Bras 2 (rouge)\n"
    f"   Temps    : {results['Bras 2']['time']:.1f} s ({results['Bras 2']['pct']:.1f}%)\n"
    f"   Entrées  : {results['Bras 2']['entries']}\n\n"
    f"Bras 3 (vert)\n"
    f"   Temps    : {results['Bras 3']['time']:.1f} s ({results['Bras 3']['pct']:.1f}%)\n"
    f"   Entrées  : {results['Bras 3']['entries']}\n\n"
    f"Séquence totale     : {len(arm_sequence)} visites\n"
    f"Alternances         : {alternations}\n"
    f"% Alternance        : {pct_alternance:.1f}%\n\n"
    f"Lissage gaussien    : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation depuis points fiables"
)
ax2.text(0.02, 0.02, stats_text,
         transform=ax2.transAxes,
         fontsize=8, color='white',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Analyse Y-Maze — couverture optimale — Video_47', color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé Y-Maze (couverture optimale) :")
print(f"   Durée totale          : {total_time_video:.1f} s")
print(f"   Fiabilité brute nose  : {pct_reliable:.1f}%")
for nom in noms:
    print(f"   {nom} → Temps : {results[nom]['time']:.1f} s | Entrées : {results[nom]['entries']}")
print(f"   Séquence totale       : {len(arm_sequence)} visites")
print(f"   Alternances           : {alternations}")
print(f"   % Alternance          : {pct_alternance:.1f}%")
