import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_38DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
IMG_PATH = r"C:\Users\Behaviour\Desktop\img00058.png"
OUTPUT   = r"C:\Users\Behaviour\Desktop\NOR_optimal.png"

# Objets (x, y, rayon)
OBJET1 = {'cx': 865,  'cy': 722, 'r': 44, 'color': '#00d4ff', 'nom': 'Objet 1'}
OBJET2 = {'cx': 1187, 'cy': 412, 'r': 42, 'color': '#ff6b6b', 'nom': 'Objet 2'}

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99

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

# ── TEMPS DANS CHAQUE OBJET ──────────────────────────────────────────────────
def time_in_circle(cx, cy, r, x, y):
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    in_c = dist <= r
    return np.sum(in_c), np.sum(in_c) / FPS

frames_o1, time_o1 = time_in_circle(OBJET1['cx'], OBJET1['cy'], OBJET1['r'], xv, yv)
frames_o2, time_o2 = time_in_circle(OBJET2['cx'], OBJET2['cy'], OBJET2['r'], xv, yv)
pct_o1 = frames_o1 / total_frames_video * 100
pct_o2 = frames_o2 / total_frames_video * 100

total_exploration = time_o1 + time_o2
DI = (time_o2 - time_o1) / total_exploration * 100 if total_exploration > 0 else 0

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')

ax1 = axes[0]
try:
    img = np.array(Image.open(IMG_PATH))
    ax1.imshow(img, alpha=0.55)
except Exception:
    ax1.set_facecolor('#0f0f23')

n    = len(xv)
cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
for i in range(n - 1):
    t = i / n
    ax1.plot([xv[i], xv[i+1]], [yv[i], yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.7)

ax1.scatter(xv[0],  yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(xv[-1], yv[-1], color='red',  s=80, zorder=5, label='Fin')

circle1_fill = plt.Circle((OBJET1['cx'], OBJET1['cy']), OBJET1['r'],
                           color=OBJET1['color'], alpha=0.2)
circle1_border = plt.Circle((OBJET1['cx'], OBJET1['cy']), OBJET1['r'],
                             color=OBJET1['color'], fill=False, linewidth=2.5,
                             label=f"{OBJET1['nom']} ({time_o1:.1f}s)")
ax1.add_patch(circle1_fill)
ax1.add_patch(circle1_border)
ax1.scatter(OBJET1['cx'], OBJET1['cy'], color=OBJET1['color'], s=50, zorder=6)
ax1.annotate(OBJET1['nom'], (OBJET1['cx'], OBJET1['cy']),
             color=OBJET1['color'], fontsize=9, fontweight='bold',
             xytext=(0, OBJET1['r']+10), textcoords='offset points', ha='center')

circle2_fill = plt.Circle((OBJET2['cx'], OBJET2['cy']), OBJET2['r'],
                           color=OBJET2['color'], alpha=0.2)
circle2_border = plt.Circle((OBJET2['cx'], OBJET2['cy']), OBJET2['r'],
                             color=OBJET2['color'], fill=False, linewidth=2.5,
                             label=f"{OBJET2['nom']} ({time_o2:.1f}s)")
ax1.add_patch(circle2_fill)
ax1.add_patch(circle2_border)
ax1.scatter(OBJET2['cx'], OBJET2['cy'], color=OBJET2['color'], s=50, zorder=6)
ax1.annotate(OBJET2['nom'], (OBJET2['cx'], OBJET2['cy']),
             color=OBJET2['color'], fontsize=9, fontweight='bold',
             xytext=(0, OBJET2['r']+10), textcoords='offset points', ha='center')

ax1.set_title('Trajectoire & exploration — couverture optimale (nose)', color='white', fontsize=13, pad=10)
ax1.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', labelcolor='white')
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

bars = ax2.bar(
    [OBJET1['nom'], OBJET2['nom']],
    [time_o1, time_o2],
    color=[OBJET1['color'], OBJET2['color']],
    width=0.5, alpha=0.85, edgecolor='white', linewidth=1
)
for bar, val in zip(bars, [time_o1, time_o2]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f} s', ha='center', va='bottom',
             color='white', fontsize=12, fontweight='bold')

ax2.set_ylabel('Temps d\'exploration (s)', color='white', fontsize=11)
ax2.set_title('Comparaison exploration des objets', color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

stats_text = (
    f"STATISTIQUES NOR — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD}) : {pct_reliable:.1f}%\n\n"
    f"Objet 1 (bleu)\n"
    f"   Temps exploration : {time_o1:.1f} s ({pct_o1:.1f}%)\n\n"
    f"Objet 2 (rouge)\n"
    f"   Temps exploration : {time_o2:.1f} s ({pct_o2:.1f}%)\n\n"
    f"Index discrimination : {DI:.1f}%\n"
    f"   (>0 = préférence Objet 2)\n\n"
    f"Lissage gaussien : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation depuis points fiables"
)
ax2.text(0.02, 0.02, stats_text,
         transform=ax2.transAxes,
         fontsize=9, color='white',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Analyse NOR — couverture optimale — Video_38', color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé NOR (couverture optimale) :")
print(f"   Durée totale vidéo    : {total_time_video:.1f} s")
print(f"   Fiabilité brute nose  : {pct_reliable:.1f}%")
print(f"   Objet 1 exploration   : {time_o1:.1f} s ({pct_o1:.1f}%)")
print(f"   Objet 2 exploration   : {time_o2:.1f} s ({pct_o2:.1f}%)")
print(f"   Index discrimination  : {DI:.1f}%")
