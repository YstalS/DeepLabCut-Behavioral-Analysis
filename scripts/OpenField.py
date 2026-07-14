import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from PIL import Image

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_29DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
IMG_PATH = r"C:\Users\Behaviour\Desktop\img00441.png"
OUTPUT   = r"C:\Users\Behaviour\Desktop\29.png"

# Coins du carré central (x, y)
CENTER_SQUARE = np.array([
    [1003,  455],   # haut gauche
    [1276, 455],   # haut droite
    [1276, 708],   # bas droite
    [1003,  708],   # bas gauche
])

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99

# ── CHARGEMENT CSV ───────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH, header=[0,1,2], index_col=0)
scorer = df.columns[0][0]

nose_x = df[(scorer, "nose", "x")].values.astype(float)
nose_y = df[(scorer, "nose", "y")].values.astype(float)
nose_p = df[(scorer, "nose", "likelihood")].values.astype(float)

tailbase_x = df[(scorer, "tail_base", "x")].values.astype(float)
tailbase_y = df[(scorer, "tail_base", "y")].values.astype(float)
tailbase_p = df[(scorer, "tail_base", "likelihood")].values.astype(float)

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

reliable_nose     = nose_p >= LIKELIHOOD_THRESHOLD
reliable_tailbase = tailbase_p >= LIKELIHOOD_THRESHOLD

pct_reliable_nose     = reliable_nose.sum()     / total_frames_video * 100
pct_reliable_tailbase = reliable_tailbase.sum() / total_frames_video * 100

nose_x_interp = interp_from_anchors(nose_x, reliable_nose)
nose_y_interp = interp_from_anchors(nose_y, reliable_nose)
nose_x_smooth = gaussian_filter1d(nose_x_interp, sigma=GAUSSIAN_SIGMA)
nose_y_smooth = gaussian_filter1d(nose_y_interp, sigma=GAUSSIAN_SIGMA)

tailbase_x_interp = interp_from_anchors(tailbase_x, reliable_tailbase)
tailbase_y_interp = interp_from_anchors(tailbase_y, reliable_tailbase)
tailbase_x_smooth = gaussian_filter1d(tailbase_x_interp, sigma=GAUSSIAN_SIGMA)
tailbase_y_smooth = gaussian_filter1d(tailbase_y_interp, sigma=GAUSSIAN_SIGMA)

frames_detected = total_frames_video
pct_detected    = 100.0

# ── TEMPS DANS LE CARRÉ CENTRAL (nez OU base de la queue) ────────────────────
# Un frame compte comme "dans le carré" si le NEZ ou la BASE DE LA QUEUE s'y
# trouve. Ça évite de sous-estimer le temps quand le rat a le corps dans le
# carré mais la tête tournée vers l'extérieur (nez hors du carré, queue dedans).
square_path = Path(np.vstack([CENTER_SQUARE, CENTER_SQUARE[0]]))

points_nose     = np.column_stack([nose_x_smooth, nose_y_smooth])
points_tailbase = np.column_stack([tailbase_x_smooth, tailbase_y_smooth])

in_square_nose     = square_path.contains_points(points_nose)
in_square_tailbase = square_path.contains_points(points_tailbase)
in_square_combined = in_square_nose | in_square_tailbase   # OR logique

frames_in_square_nose     = np.sum(in_square_nose)
frames_in_square_tailbase = np.sum(in_square_tailbase)
frames_in_square_combined = np.sum(in_square_combined)

time_in_square_nose     = frames_in_square_nose     / FPS
time_in_square_tailbase = frames_in_square_tailbase / FPS
time_in_square_combined = frames_in_square_combined / FPS

pct_square_nose     = frames_in_square_nose     / total_frames_video * 100
pct_square_tailbase = frames_in_square_tailbase / total_frames_video * 100
pct_square_combined = frames_in_square_combined / total_frames_video * 100

# Métrique retenue pour l'affichage principal = combinée (nez OU base queue)
in_square        = in_square_combined
frames_in_square = frames_in_square_combined
time_in_square   = time_in_square_combined
pct_square       = pct_square_combined

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor('#1a1a2e')

# ── SUBPLOT 1 : Trajectoire ───────────────────────────────────────────────────
ax1 = axes[0]
try:
    img = np.array(Image.open(IMG_PATH))
    ax1.imshow(img, alpha=0.55)
except Exception:
    ax1.set_facecolor('#0f0f23')

xv = nose_x_smooth
yv = nose_y_smooth
n  = len(xv)

cmap = LinearSegmentedColormap.from_list("traj", ["#00d4ff", "#7b2ff7", "#ff6b6b"])
for i in range(n - 1):
    t = i / n
    ax1.plot([xv[i], xv[i+1]], [yv[i], yv[i+1]],
             color=cmap(t), linewidth=0.9, alpha=0.75)

ax1.scatter(xv[0],  yv[0],  color='lime', s=80, zorder=5, label='Début')
ax1.scatter(xv[-1], yv[-1], color='red',  s=80, zorder=5, label='Fin')

square_fill = patches.Polygon(
    CENTER_SQUARE, closed=True,
    linewidth=2, edgecolor='yellow',
    facecolor='yellow', alpha=0.15, linestyle='--'
)
ax1.add_patch(square_fill)
square_border = patches.Polygon(
    CENTER_SQUARE, closed=True,
    linewidth=2, edgecolor='yellow',
    facecolor='none', linestyle='--', label='Carré central'
)
ax1.add_patch(square_border)

labels_pos = [('HG', CENTER_SQUARE[0]), ('HD', CENTER_SQUARE[1]),
              ('BD', CENTER_SQUARE[2]), ('BG', CENTER_SQUARE[3])]
for lbl, (px, py) in labels_pos:
    ax1.scatter(px, py, color='white', s=30, zorder=6)
    ax1.annotate(lbl, (px, py), color='white', fontsize=7,
                 xytext=(4, 4), textcoords='offset points')

ax1.set_title('Trajectoire — couverture optimale (nose)', color='white', fontsize=13, pad=10)
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

# ── SUBPLOT 2 : Heatmap + stats ───────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0f0f23')

xmin, xmax = 800, 1400
ymin, ymax = 400, 850

h, _, _ = np.histogram2d(
    nose_x_smooth, nose_y_smooth,
    bins=60,
    range=[[xmin, xmax], [ymin, ymax]]
)
h_smooth = gaussian_filter(h.T, sigma=2)
im = ax2.imshow(h_smooth, origin='upper',
                extent=[xmin, xmax, ymax, ymin],
                cmap='inferno', aspect='auto')

square_hm = patches.Polygon(
    CENTER_SQUARE, closed=True,
    linewidth=2, edgecolor='yellow',
    facecolor='none', linestyle='--'
)
ax2.add_patch(square_hm)

cbar2 = fig.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
cbar2.set_label('Fréquence', color='white', fontsize=9)
cbar2.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

stats_text = (
    f"STATISTIQUES — COUVERTURE OPTIMALE\n\n"
    f"Durée totale vidéo   : {total_time_video:.1f} s\n"
    f"Frames totales       : {total_frames_video:,}\n"
    f"Couverture analysée  : {pct_detected:.0f}% (100% — interpolée)\n"
    f"Fiabilité brute (likelihood≥{LIKELIHOOD_THRESHOLD})\n"
    f"   nez        : {pct_reliable_nose:.1f}%\n"
    f"   base queue : {pct_reliable_tailbase:.1f}%\n\n"
    f"Temps carré central — nez OU base queue : {time_in_square_combined:.1f} s\n"
    f"   ({pct_square_combined:.1f}% du temps total)\n"
    f"   -> nez seul        : {time_in_square_nose:.1f} s ({pct_square_nose:.1f}%)\n"
    f"   -> base queue seule: {time_in_square_tailbase:.1f} s ({pct_square_tailbase:.1f}%)\n\n"
    f"Lissage gaussien : sigma={GAUSSIAN_SIGMA}\n"
    f"Méthode : interpolation depuis points fiables (nez + base queue)"
)
ax2.text(0.02, 0.02, stats_text,
         transform=ax2.transAxes,
         fontsize=9, color='white',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

ax2.set_title('Heatmap de présence', color='white', fontsize=13, pad=10)
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

plt.suptitle('Analyse Open Field — couverture optimale — Video_29', color='white', fontsize=15)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé Open Field (couverture optimale) :")
print(f"   Durée totale vidéo               : {total_time_video:.1f} s")
print(f"   Fiabilité brute nez               : {pct_reliable_nose:.1f}%")
print(f"   Fiabilité brute base queue        : {pct_reliable_tailbase:.1f}%")
print(f"   Temps carré central (nez ou queue): {time_in_square_combined:.1f} s ({pct_square_combined:.1f}% du temps total)")
print(f"      dont nez seul                 : {time_in_square_nose:.1f} s ({pct_square_nose:.1f}%)")
print(f"      dont base queue seule         : {time_in_square_tailbase:.1f} s ({pct_square_tailbase:.1f}%)")