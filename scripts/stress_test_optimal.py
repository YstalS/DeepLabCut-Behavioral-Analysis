import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\PC\Desktop\Trial     2DLC_Resnet50_pfe trainingApr11shuffle1_snapshot_best-50_filtered.csv"
IMG_PATH = r"C:\Users\PC\Desktop\img00191.png"
OUTPUT   = r"C:\Users\PC\Desktop\stress_test_optimal.png"

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99

# ── ZONES (x, y) — conversion depuis Y X ─────────────────────────────────────
ZONE_CONFORT = np.array([
    [47,  53],
    [316, 53],
    [316, 323],
    [47,  323],
])

ZONE_STRESS = np.array([
    [316, 53],
    [550, 53],
    [550, 323],
    [316, 323],
])

PETRI_CENTER = (454, 184)
PETRI_EDGE   = (425, 184)
PETRI_RADIUS = abs(PETRI_CENTER[0] - PETRI_EDGE[0])  # 29 px

ZONES = {
    'Zone confort': {'poly': ZONE_CONFORT, 'color': '#00d4ff'},
    'Zone stress':  {'poly': ZONE_STRESS,  'color': '#ff6b6b'},
}

# ── CHARGEMENT CSV ───────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH, header=[0, 1, 2], index_col=0)
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

# ── TEMPS DANS CHAQUE ZONE (rectangles) ──────────────────────────────────────
def analyze_zone_poly(poly, x, y):
    path    = Path(np.vstack([poly, poly[0]]))
    points  = np.column_stack([x, y])
    in_zone = path.contains_points(points)
    frames_z = np.sum(in_zone)
    time_z   = frames_z / FPS
    pct_z    = frames_z / total_frames_video * 100
    return time_z, pct_z

results = {}
for nom, zone in ZONES.items():
    time_z, pct_z = analyze_zone_poly(zone['poly'], xv, yv)
    results[nom] = {'time': time_z, 'pct': pct_z}

# ── INTERACTIONS AVEC LA BOÎTE DE PÉTRI (comme dans NOR) ────────────────────
dist_to_petri  = np.sqrt((xv - PETRI_CENTER[0])**2 + (yv - PETRI_CENTER[1])**2)
in_petri       = dist_to_petri <= PETRI_RADIUS
frames_petri   = np.sum(in_petri)
time_petri     = frames_petri / FPS
pct_petri      = frames_petri / total_frames_video * 100
interactions_petri = np.sum(np.diff(in_petri.astype(int)) == 1)

# ── FIGURE (2x2) ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.patch.set_facecolor('#1a1a2e')
ax1, ax2 = axes[0]
ax3, ax4 = axes[1]

# ── AX1 : Trajectoire + zones + boîte de pétri ───────────────────────────────
try:
    img = np.array(Image.open(IMG_PATH))
    ax1.imshow(img, alpha=0.55)
except Exception:
    ax1.set_facecolor('#0f0f23')

for nom, zone in ZONES.items():
    poly_fill   = patches.Polygon(zone['poly'], closed=True,
                                  facecolor=zone['color'], alpha=0.18, linewidth=0)
    poly_border = patches.Polygon(zone['poly'], closed=True,
                                  edgecolor=zone['color'], facecolor='none',
                                  linewidth=2.5, linestyle='--',
                                  label=f"{nom} ({results[nom]['time']:.1f}s | {results[nom]['pct']:.1f}%)")
    ax1.add_patch(poly_fill)
    ax1.add_patch(poly_border)
    cx = np.mean(zone['poly'][:, 0])
    cy = np.mean(zone['poly'][:, 1])
    ax1.text(cx, cy, nom, color=zone['color'], fontsize=10,
             ha='center', va='center', fontweight='bold')

petri_fill   = plt.Circle(PETRI_CENTER, PETRI_RADIUS, color='#ffd93d', alpha=0.25)
petri_border = plt.Circle(PETRI_CENTER, PETRI_RADIUS, color='#ffd93d', fill=False,
                           linewidth=2.5,
                           label=f"Boîte de pétri ({time_petri:.1f}s | {interactions_petri} interactions)")
ax1.add_patch(petri_fill)
ax1.add_patch(petri_border)
ax1.scatter(*PETRI_CENTER, color='#ffd93d', s=40, zorder=6)
ax1.annotate('Pétri', PETRI_CENTER, color='#ffd93d', fontsize=9, fontweight='bold',
             xytext=(0, PETRI_RADIUS + 10), textcoords='offset points', ha='center')

n    = len(xv)
cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
for i in range(n - 1):
    t = i / n
    ax1.plot([xv[i], xv[i+1]], [yv[i], yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.7)

ax1.scatter(xv[0],  yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(xv[-1], yv[-1], color='red',  s=80, zorder=5, label='Fin')

ax1.set_title('Trajectoire — couverture optimale (nose)', color='white', fontsize=13, pad=10)
ax1.legend(loc='upper right', fontsize=7.5, facecolor='#1a1a2e', labelcolor='white')
try:
    img_h, img_w = img.shape[:2]
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
except Exception:
    ax1.set_xlim(0, 600)
    ax1.set_ylim(380, 0)
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, total_time_video))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', fraction=0.03, pad=0.02)
cbar.set_label('Temps (s)', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# ── AX2 : Barres temps par zone ───────────────────────────────────────────────
ax2.set_facecolor('#0f0f23')
noms_zones   = list(ZONES.keys())
temps_zones  = [results[n]['time'] for n in noms_zones]
colors_zones = [ZONES[n]['color']  for n in noms_zones]

bars = ax2.bar(noms_zones, temps_zones, color=colors_zones, width=0.5,
               alpha=0.85, edgecolor='white', linewidth=1)
for bar, val in zip(bars, temps_zones):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f} s', ha='center', va='bottom',
             color='white', fontsize=12, fontweight='bold')

ax2.set_ylabel("Temps passé (s)", color='white', fontsize=11)
ax2.set_title('Temps passé par zone', color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# ── AX3 : Boîte de pétri — temps + interactions ──────────────────────────────
ax3.set_facecolor('#0f0f23')

ax3.bar(['Temps\nd\'exploration'], [time_petri], color='#ffd93d',
        width=0.4, alpha=0.85, edgecolor='white', linewidth=1)
ax3.text(0, time_petri + max(time_petri*0.03, 0.2), f'{time_petri:.1f} s',
          ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax3.set_ylabel('Temps (s)', color='#ffd93d', fontsize=11)
ax3.tick_params(axis='y', colors='#ffd93d')
ax3.tick_params(axis='x', colors='white')

ax3b = ax3.twinx()
ax3b.set_facecolor('#0f0f23')
ax3b.bar(['Nombre\nd\'interactions'], [interactions_petri], color='#ff9f43',
         width=0.4, alpha=0.85, edgecolor='white', linewidth=1)
ax3b.text(1, interactions_petri + max(interactions_petri*0.03, 0.2), f'{interactions_petri}',
           ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
ax3b.set_ylabel('Nombre d\'interactions', color='#ff9f43', fontsize=11)
ax3b.tick_params(axis='y', colors='#ff9f43')

ax3.set_xlim(-0.5, 1.5)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Temps\nd\'exploration', 'Nombre\nd\'interactions'], color='white')
ax3.set_title('Boîte de pétri — Temps & interactions', color='white', fontsize=13, pad=10)
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')
for spine in ax3b.spines.values():
    spine.set_edgecolor('#444')

# ── AX4 : Stats texte ────────────────────────────────────────────────────────
ax4.set_facecolor('#0f0f23')
ax4.axis('off')

stats_text = (
    f"STATISTIQUES — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD}) : {pct_reliable:.1f}%\n\n"
    f"Zone confort\n"
    f"   Temps : {results['Zone confort']['time']:.1f} s ({results['Zone confort']['pct']:.1f}%)\n\n"
    f"Zone stress\n"
    f"   Temps : {results['Zone stress']['time']:.1f} s ({results['Zone stress']['pct']:.1f}%)\n\n"
    f"Boîte de pétri (dans la zone stress)\n"
    f"   Temps d'exploration : {time_petri:.1f} s ({pct_petri:.1f}%)\n"
    f"   Nombre d'interactions : {interactions_petri}\n\n"
    f"Lissage gaussien : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation à partir des points fiables\n"
    f"          (pas de frames supprimées)"
)
ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
          fontsize=10, color='white', verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Analyse Stress Test — couverture optimale', color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')

print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé Stress Test (couverture optimale) :")
print(f"   Durée totale vidéo     : {total_time_video:.1f} s")
print(f"   Fiabilité brute nose   : {pct_reliable:.1f}%")
print(f"   Zone confort           : {results['Zone confort']['time']:.1f} s ({results['Zone confort']['pct']:.1f}%)")
print(f"   Zone stress            : {results['Zone stress']['time']:.1f} s ({results['Zone stress']['pct']:.1f}%)")
print(f"   Boîte de pétri - temps        : {time_petri:.1f} s ({pct_petri:.1f}%)")
print(f"   Boîte de pétri - interactions : {interactions_petri}")
