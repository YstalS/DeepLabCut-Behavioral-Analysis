import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ── PARAMETRES ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Behaviour\Desktop\Video_29DLC_Resnet50_finalsMay29shuffle1_snapshot_best-100.csv"
OUTPUT   = r"C:\Users\Behaviour\Desktop\29-distance.png"

LIKELIHOOD_THRESHOLD = 0.6   # seuil pour choisir les points FIABLES comme ancres
GAUSSIAN_SIGMA        = 5
FPS                   = 29.99
CM_PER_PIXEL          = 0.119  # 46cm / 387px

# ── CHARGEMENT CSV ───────────────────────────────────────────────────────────
df     = pd.read_csv(CSV_PATH, header=[0,1,2], index_col=0)
scorer = df.columns[0][0]

nose_x = df[(scorer, "nose", "x")].values.astype(float)
nose_y = df[(scorer, "nose", "y")].values.astype(float)
nose_p = df[(scorer, "nose", "likelihood")].values.astype(float)

total_time_video = len(nose_x) / FPS

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
pct_reliable = reliable.sum() / len(nose_p) * 100

nose_x_interp = interp_from_anchors(nose_x, reliable)
nose_y_interp = interp_from_anchors(nose_y, reliable)
nose_x_smooth = gaussian_filter1d(nose_x_interp, sigma=GAUSSIAN_SIGMA)
nose_y_smooth = gaussian_filter1d(nose_y_interp, sigma=GAUSSIAN_SIGMA)

# ── CALCUL DISTANCE ET VITESSE ───────────────────────────────────────────────
dx = np.diff(nose_x_smooth)
dy = np.diff(nose_y_smooth)

dist_per_frame = np.sqrt(dx**2 + dy**2) * CM_PER_PIXEL

dist_cumulative = np.cumsum(dist_per_frame)
total_distance  = dist_cumulative[-1]

speed = dist_per_frame * FPS  # cm/frame * frames/s = cm/s

time_axis = np.arange(len(nose_x)) / FPS

speed_smooth = gaussian_filter1d(speed, sigma=10)

valid_speed  = speed[speed > 0]
mean_speed   = np.mean(valid_speed)
max_speed    = np.max(valid_speed)
median_speed = np.median(valid_speed)

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.patch.set_facecolor('#1a1a2e')

colors = {'speed': '#00d4ff', 'dist': '#ff6b6b', 'hist': '#7b2ff7'}

# ── SUBPLOT 1 : Vitesse au fil du temps ──────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor('#0f0f23')
ax1.plot(time_axis[:-1], speed_smooth, color=colors['speed'], linewidth=1, alpha=0.9)
ax1.axhline(mean_speed, color='yellow', linewidth=1.5, linestyle='--', label=f'Moyenne : {mean_speed:.1f} cm/s')
ax1.axhline(max_speed,  color='red',    linewidth=1.5, linestyle='--', label=f'Max : {max_speed:.1f} cm/s')
ax1.set_xlabel('Temps (s)', color='white', fontsize=10)
ax1.set_ylabel('Vitesse (cm/s)', color='white', fontsize=10)
ax1.set_title('Vitesse du rat au fil du temps — couverture optimale', color='white', fontsize=12, pad=8)
ax1.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
ax1.tick_params(colors='white')
ax1.set_xlim(0, total_time_video)
for spine in ax1.spines.values():
    spine.set_edgecolor('#444')

# ── SUBPLOT 2 : Distance cumulée ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0f0f23')
ax2.plot(time_axis[:-1], dist_cumulative, color=colors['dist'], linewidth=1.5)
ax2.set_xlabel('Temps (s)', color='white', fontsize=10)
ax2.set_ylabel('Distance cumulée (cm)', color='white', fontsize=10)
ax2.set_title(f'Distance totale parcourue : {total_distance:.1f} cm  ({total_distance/100:.2f} m)', color='white', fontsize=12, pad=8)
ax2.tick_params(colors='white')
ax2.set_xlim(0, total_time_video)
for spine in ax2.spines.values():
    spine.set_edgecolor('#444')

# ── SUBPLOT 3 : Distribution des vitesses ────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor('#0f0f23')
ax3.hist(valid_speed, bins=60, color=colors['hist'], alpha=0.8, edgecolor='none')
ax3.axvline(mean_speed,   color='yellow', linewidth=2, linestyle='--', label=f'Moyenne : {mean_speed:.1f} cm/s')
ax3.axvline(median_speed, color='cyan',   linewidth=2, linestyle='--', label=f'Médiane : {median_speed:.1f} cm/s')
ax3.axvline(max_speed,    color='red',    linewidth=2, linestyle='--', label=f'Max : {max_speed:.1f} cm/s')
ax3.set_xlabel('Vitesse (cm/s)', color='white', fontsize=10)
ax3.set_ylabel('Nombre de frames', color='white', fontsize=10)
ax3.set_title('Distribution des vitesses', color='white', fontsize=12, pad=8)
ax3.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')

plt.suptitle(f'Analyse vitesse & distance — couverture optimale-29',
             color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')

print(f"Graphe sauvegardé : {OUTPUT}")
print(f"\nRésumé (couverture optimale) :")
print(f"   Durée totale vidéo   : {total_time_video:.1f} s")
print(f"   Fiabilité brute nose : {pct_reliable:.1f}%")
print(f"   Distance totale      : {total_distance:.1f} cm ({total_distance/100:.2f} m)")
print(f"   Vitesse moyenne      : {mean_speed:.1f} cm/s")
print(f"   Vitesse médiane      : {median_speed:.1f} cm/s")
print(f"   Vitesse maximale     : {max_speed:.1f} cm/s")
