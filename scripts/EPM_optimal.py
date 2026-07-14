import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_61DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
IMG_PATH = r"C:\Users\Behaviour\Desktop\img00000.png"
OUTPUT   = r"C:\Users\Behaviour\Desktop\EPM_optimal.png"

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99

# ── ZONES EPM (x, y) ─────────────────────────────────────────────────────────
CENTRE = np.array([
    [874, 566], [962, 566], [962, 650], [874, 650],
])
BRAS_OUVERT_1 = np.array([
    [874, 566], [874, 650], [431, 665], [431, 580],
])
BRAS_OUVERT_2 = np.array([
    [962, 566], [962, 650], [1360, 629], [1356, 553],
])
BRAS_FERME_1 = np.array([
    [874, 566], [962, 566], [946, 181], [886, 181],
])
BRAS_FERME_2 = np.array([
    [874, 659], [962, 650], [1027, 1072], [901, 1072],
])

ZONES = {
    'Centre':        {'poly': CENTRE,        'color': '#ffffff'},
    'Bras ouvert 1': {'poly': BRAS_OUVERT_1, 'color': '#00d4ff'},
    'Bras ouvert 2': {'poly': BRAS_OUVERT_2, 'color': '#00ffaa'},
    'Bras fermé 1':  {'poly': BRAS_FERME_1,  'color': '#ff6b6b'},
    'Bras fermé 2':  {'poly': BRAS_FERME_2,  'color': '#ff9f43'},
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

# ── TEMPS + ENTRÉES DANS CHAQUE ZONE ─────────────────────────────────────────
def analyze_zone(poly, x, y):
    path    = Path(np.vstack([poly, poly[0]]))
    points  = np.column_stack([x, y])
    in_zone = path.contains_points(points)
    frames_z = np.sum(in_zone)
    time_z   = frames_z / FPS
    pct_z    = frames_z / total_frames_video * 100
    entries  = np.sum(np.diff(in_zone.astype(int)) == 1)
    return time_z, pct_z, entries

results = {}
for nom, zone in ZONES.items():
    time_z, pct_z, entries = analyze_zone(zone['poly'], xv, yv)
    results[nom] = {'time': time_z, 'pct': pct_z, 'entries': entries}

total_open   = results['Bras ouvert 1']['time'] + results['Bras ouvert 2']['time']
total_closed = results['Bras fermé 1']['time']  + results['Bras fermé 2']['time']
total_arms   = total_open + total_closed
anxiety_index = (total_closed - total_open) / total_arms * 100 if total_arms > 0 else 0

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
                                  linewidth=2, linestyle='--',
                                  label=f"{nom} ({results[nom]['time']:.1f}s | {results[nom]['entries']} entrées)")
    ax1.add_patch(poly_fill)
    ax1.add_patch(poly_border)
    cx = np.mean(zone['poly'][:, 0])
    cy = np.mean(zone['poly'][:, 1])
    ax1.text(cx, cy, nom.replace(' ', '\n'), color=zone['color'],
             fontsize=7, ha='center', va='center', fontweight='bold')

n    = len(xv)
cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
for i in range(n - 1):
    t = i / n
    ax1.plot([xv[i], xv[i+1]], [yv[i], yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.75)

ax1.scatter(xv[0],  yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(xv[-1], yv[-1], color='red',  s=80, zorder=5, label='Fin')

ax1.set_title('Trajectoire EPM — couverture optimale (nose)', color='white', fontsize=13, pad=10)
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

noms    = list(results.keys())
temps   = [results[n]['time']    for n in noms]
entrees = [results[n]['entries'] for n in noms]
colors  = [ZONES[n]['color']     for n in noms]

x_pos = np.arange(len(noms))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, temps, width,
                color=colors, alpha=0.85, edgecolor='white', linewidth=1, label='Temps (s)')
for bar, val in zip(bars1, temps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}s', ha='center', va='bottom',
             color='white', fontsize=9, fontweight='bold')

ax2b = ax2.twinx()
ax2b.set_facecolor('#0f0f23')
bars2 = ax2b.bar(x_pos + width/2, entrees, width,
                 color=colors, alpha=0.45, edgecolor='white',
                 linewidth=1, label='Entrées')
for bar, val in zip(bars2, entrees):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f'{val}', ha='center', va='bottom',
              color='white', fontsize=9, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(noms, color='white', fontsize=8)
plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
ax2.set_ylabel('Temps (s)', color='white', fontsize=10)
ax2b.set_ylabel('Nombre d\'entrées', color='white', fontsize=10)
ax2.set_title('Temps & entrées par zone', color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
ax2b.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

ax2.legend(['Temps (barres pleines)', 'Entrées (barres transparentes)'],
           fontsize=8, facecolor='#1a1a2e', labelcolor='white', loc='upper right')

stats_text = (
    f"STATISTIQUES EPM — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD}) : {pct_reliable:.1f}%\n\n"
    f"Centre              : {results['Centre']['time']:.1f} s | {results['Centre']['entries']} entrées\n\n"
    f"Bras ouverts\n"
    f"   Bras ouvert 1    : {results['Bras ouvert 1']['time']:.1f} s | {results['Bras ouvert 1']['entries']} entrées\n"
    f"   Bras ouvert 2    : {results['Bras ouvert 2']['time']:.1f} s | {results['Bras ouvert 2']['entries']} entrées\n"
    f"   Total ouverts    : {total_open:.1f} s\n\n"
    f"Bras fermés\n"
    f"   Bras fermé 1     : {results['Bras fermé 1']['time']:.1f} s | {results['Bras fermé 1']['entries']} entrées\n"
    f"   Bras fermé 2     : {results['Bras fermé 2']['time']:.1f} s | {results['Bras fermé 2']['entries']} entrées\n"
    f"   Total fermés     : {total_closed:.1f} s\n\n"
    f"Index anxiété       : {anxiety_index:.1f}%\n"
    f"   (>0 = anxieux)\n\n"
    f"Lissage gaussien    : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation depuis points fiables"
)
ax2.text(0.02, 0.02, stats_text,
         transform=ax2.transAxes,
         fontsize=8, color='white',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Analyse EPM — couverture optimale — Video_61', color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé EPM (couverture optimale) :")
print(f"   Durée totale vidéo  : {total_time_video:.1f} s")
print(f"   Fiabilité brute nose : {pct_reliable:.1f}%")
for nom in noms:
    print(f"   {nom:20s} : {results[nom]['time']:.1f} s ({results[nom]['pct']:.1f}%) | {results[nom]['entries']} entrées")
print(f"   Total bras ouverts  : {total_open:.1f} s")
print(f"   Total bras fermés   : {total_closed:.1f} s")
print(f"   Index anxiété       : {anxiety_index:.1f}%")
