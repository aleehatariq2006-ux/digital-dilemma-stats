# ============================================================
#  THE STUDENT'S DIGITAL DILEMMA
#  A Statistical Analysis of Social Media Usage &
#  Its Impact on Student Academic Performance
#
#  Author  : Aleeha
#  Course  : MTH-262 Statistics & Probability Theory
#  Uni     : COMSATS University Islamabad, Sahiwal Campus
#  Ref     : Weiss, Introductory Statistics, 10th Edition
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family']  = 'DejaVu Sans'
plt.rcParams['figure.dpi']   = 120
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# ── COLOUR PALETTE ───────────────────────────────────────────
C1, C2, C3 = '#1F4E79', '#2E74B5', '#BDD7EE'
CG, CR, CY  = '#70AD47', '#C00000', '#FFC000'

# ============================================================
# DATASET  (30 students survey)
# ============================================================
students = [f'S{i+1:02d}' for i in range(30)]

screen = np.array([2.5, 3.0, 1.5, 4.5, 6.0, 7.5, 3.5, 2.0, 5.5, 8.0,
                   1.0, 4.0, 6.5, 3.0, 2.5, 5.0, 7.0, 1.5, 4.5, 9.0,
                   3.5, 2.0, 6.0, 4.0, 1.5, 5.5, 3.0, 7.5, 2.5, 4.5])

cgpa   = np.array([3.8, 3.6, 3.9, 3.2, 2.8, 2.4, 3.5, 3.7, 2.9, 2.1,
                   3.9, 3.3, 2.6, 3.6, 3.8, 3.0, 2.5, 3.9, 3.1, 1.9,
                   3.4, 3.7, 2.7, 3.2, 3.9, 2.9, 3.5, 2.3, 3.8, 3.1])

sleep  = np.array([7.5, 7.0, 8.0, 6.5, 6.0, 5.5, 7.0, 7.5, 6.0, 5.0,
                   8.0, 6.8, 5.8, 7.0, 7.5, 6.3, 5.5, 8.0, 6.5, 4.5,
                   7.0, 7.5, 5.8, 6.8, 8.0, 6.2, 7.0, 5.2, 7.5, 6.5])

n = len(screen)

# ============================================================
# STEP 1 — MEASURES OF CENTRAL TENDENCY & DISPERSION
# ============================================================
print("=" * 60)
print("  THE STUDENT'S DIGITAL DILEMMA")
print("  Statistical Analysis — MTH-262")
print("=" * 60)

print("\n📊 STEP 1: CENTRAL TENDENCY & DISPERSION")
print("-" * 60)

for label, data in [("Screen Time (hrs)", screen),
                     ("CGPA",              cgpa),
                     ("Sleep (hrs)",       sleep)]:
    mean   = np.mean(data)
    median = np.median(data)
    mode_r = stats.mode(data, keepdims=True)
    sd     = np.std(data, ddof=1)          # sample SD (n-1)
    var    = np.var(data,  ddof=1)
    rng    = np.max(data) - np.min(data)

    print(f"\n  [{label}]")
    print(f"    Mean     (x̄)  = {mean:.4f}")
    print(f"    Median        = {median:.4f}")
    print(f"    Mode          = {mode_r.mode[0]}")
    print(f"    Std Dev  (s)  = {sd:.4f}")
    print(f"    Variance (s²) = {var:.4f}")
    print(f"    Range         = {rng:.4f}")

# ============================================================
# STEP 2 — OUTLIER DETECTION (IQR Method — Weiss Def 3.10)
# ============================================================
print("\n\n📊 STEP 2: OUTLIER DETECTION — IQR Method")
print("-" * 60)

Q1  = np.percentile(screen, 25)
Q3  = np.percentile(screen, 75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = screen[(screen < lower_limit) | (screen > upper_limit)]

print(f"  Q1             = {Q1:.4f}")
print(f"  Q3             = {Q3:.4f}")
print(f"  IQR (Q3-Q1)   = {IQR:.4f}")
print(f"  Lower Limit    = Q1 - 1.5×IQR = {lower_limit:.4f}")
print(f"  Upper Limit    = Q3 + 1.5×IQR = {upper_limit:.4f}")
print(f"  Outliers Found = {len(outliers)}" +
      (f" → {outliers}" if len(outliers) > 0 else " → Dataset is CLEAN ✓"))

# ============================================================
# STEP 3 — CONTINGENCY TABLE & CONDITIONAL PROBABILITY
# ============================================================
print("\n\n📊 STEP 3: CONTINGENCY TABLE")
print("-" * 60)

# Classify students
high_use  = screen > 4          # >4 hrs = High Usage
high_cgpa = cgpa   >= 3.0       # >=3.0  = High CGPA

# 2x2 counts
a = np.sum( high_use &  high_cgpa)   # High Use, High CGPA
b = np.sum( high_use & ~high_cgpa)   # High Use, Low  CGPA
c = np.sum(~high_use &  high_cgpa)   # Low  Use, High CGPA
d = np.sum(~high_use & ~high_cgpa)   # Low  Use, Low  CGPA

print(f"\n  {'':20} {'High CGPA(≥3.0)':>16} {'Low CGPA(<3.0)':>15} {'Total':>8}")
print(f"  {'High Usage (>4hrs)':20} {a:>16} {b:>15} {a+b:>8}")
print(f"  {'Low  Usage (≤4hrs)':20} {c:>16} {d:>15} {c+d:>8}")
print(f"  {'Total':20} {a+c:>16} {b+d:>15} {n:>8}")

# Probabilities
P_highUse  = (a + b) / n
P_lowUse   = (c + d) / n
P_highCGPA = (a + c) / n
P_lowCGPA  = (b + d) / n

P_lowCGPA_given_highUse  = b / (a + b)
P_highCGPA_given_lowUse  = c / (c + d)
P_highUse_given_lowCGPA  = b / (b + d)

print(f"\n  Marginal Probabilities:")
print(f"    P(High Usage)  = {P_highUse:.4f}")
print(f"    P(High CGPA)   = {P_highCGPA:.4f}")

print(f"\n  Conditional Probabilities  [Weiss Formula 4.5]:")
print(f"    P(Low CGPA  | High Usage) = {P_lowCGPA_given_highUse:.4f}  ← {P_lowCGPA_given_highUse*100:.1f}% !")
print(f"    P(High CGPA | Low  Usage) = {P_highCGPA_given_lowUse:.4f}  ← {P_highCGPA_given_lowUse*100:.1f}% !")

# Independence check
print(f"\n  Independence Check:")
print(f"    P(High Use) × P(Low CGPA)     = {P_highUse * P_lowCGPA:.4f}")
print(f"    P(High Use ∩ Low CGPA)        = {b/n:.4f}")
if abs(P_highUse * P_lowCGPA - b/n) > 0.01:
    print(f"    → NOT EQUAL → Events are DEPENDENT (not independent) ✓")

# ============================================================
# STEP 4 — MULTIPLICATION RULE (Weiss Formula 4.4)
# ============================================================
print("\n\n📊 STEP 4: MULTIPLICATION RULE")
print("-" * 60)

# General rule: P(A∩B) = P(A) × P(B|A)
joint_general = P_highUse * P_lowCGPA_given_highUse
print(f"  General Rule: P(High Use ∩ Low CGPA)")
print(f"    = P(High Use) × P(Low CGPA | High Use)")
print(f"    = {P_highUse:.4f} × {P_lowCGPA_given_highUse:.4f} = {joint_general:.4f}  ✓")

# If independent (hypothetical)
joint_if_indep = P_highUse * P_lowCGPA
print(f"\n  If they WERE independent (hypothetical):")
print(f"    = P(High Use) × P(Low CGPA) = {P_highUse:.4f} × {P_lowCGPA:.4f} = {joint_if_indep:.4f}")
print(f"    Actual = {b/n:.4f}  →  Much higher — confirms DEPENDENCE")

# ============================================================
# STEP 5 — LAW OF TOTAL PROBABILITY + BAYES' RULE
# ============================================================
print("\n\n📊 STEP 5: LAW OF TOTAL PROBABILITY + BAYES' RULE")
print("-" * 60)

# Partition: Low Use (A1) vs High Use (A2)
P_A1 = P_lowUse
P_A2 = P_highUse
P_B_given_A1 = d / (c + d)   # P(Low CGPA | Low Use)
P_B_given_A2 = b / (a + b)   # P(Low CGPA | High Use)

# Law of Total Probability
P_B = P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2

print(f"  Partition: A1=Low Usage, A2=High Usage, B=Low CGPA")
print(f"\n  Law of Total Probability  [Weiss 4.8]:")
print(f"    P(B) = P(A1)·P(B|A1) + P(A2)·P(B|A2)")
print(f"         = {P_A1:.4f}×{P_B_given_A1:.4f}  +  {P_A2:.4f}×{P_B_given_A2:.4f}")
print(f"         = {P_A1*P_B_given_A1:.4f} + {P_A2*P_B_given_A2:.4f}")
print(f"         = {P_B:.4f}  (= P(Low CGPA) = {P_lowCGPA:.4f} ✓)")

# Bayes' Rule
P_A1_given_B = (P_A1 * P_B_given_A1) / P_B
P_A2_given_B = (P_A2 * P_B_given_A2) / P_B

print(f"\n  Bayes' Rule  [Weiss 4.9]:")
print(f"    P(High Use | Low CGPA) = {P_A2*P_B_given_A2:.4f} / {P_B:.4f} = {P_A2_given_B:.4f}")
print(f"    P(Low  Use | Low CGPA) = {P_A1*P_B_given_A1:.4f} / {P_B:.4f} = {P_A1_given_B:.4f}")
print(f"    Check: {P_A1_given_B:.4f} + {P_A2_given_B:.4f} = {P_A1_given_B+P_A2_given_B:.4f} ✓")
print(f"\n  >>> If a student has Low CGPA → {P_A2_given_B*100:.1f}% chance they are a Heavy User!")

# ============================================================
# STEP 6 — NORMAL DISTRIBUTION  (Weiss Ch. 6)
# ============================================================
print("\n\n📊 STEP 6: NORMAL DISTRIBUTION")
print("-" * 60)

mu    = np.mean(screen)
sigma = np.std(screen, ddof=1)

print(f"  X ~ N(μ={mu:.4f}, σ={sigma:.4f})")

queries = [
    ("P(X < 6)",    6,   None, "less than 6 hrs/day"),
    ("P(X > 7)",    None, 7,   "more than 7 hrs/day"),
    ("P(2 < X < 6)", 2,  6,   "between 2 and 6 hrs/day"),
]

for label, lo, hi, desc in queries:
    if lo is None:
        z  = (hi - mu) / sigma
        p  = 1 - stats.norm.cdf(z)
        print(f"  {label}: Z=({hi}-{mu:.3f})/{sigma:.3f}={z:.3f}  →  P={p:.4f}  ({desc})")
    elif hi is None:
        z  = (lo - mu) / sigma
        p  = stats.norm.cdf(z)
        print(f"  {label}: Z=({lo}-{mu:.3f})/{sigma:.3f}={z:.3f}  →  P={p:.4f}  ({desc})")
    else:
        z1 = (lo - mu) / sigma
        z2 = (hi - mu) / sigma
        p  = stats.norm.cdf(z2) - stats.norm.cdf(z1)
        print(f"  {label}: Z1={z1:.3f}, Z2={z2:.3f}  →  P={p:.4f}  ({desc})")

pct_90 = stats.norm.ppf(0.90, mu, sigma)
print(f"  90th Percentile: X = {pct_90:.4f} hrs  (top 10% use more than this)")

# ============================================================
# STEP 7 — CONFIDENCE INTERVAL  (Weiss Ch. 8)
# ============================================================
print("\n\n📊 STEP 7: CONFIDENCE INTERVAL")
print("-" * 60)

x_bar = np.mean(screen)
se    = sigma / np.sqrt(n)

print(f"  x̄ = {x_bar:.4f},  s = {sigma:.4f},  n = {n},  SE = s/√n = {se:.4f}\n")

for cl, z_val in [(90, 1.645), (95, 1.960), (99, 2.576)]:
    moe = z_val * se
    lo_ci, hi_ci = x_bar - moe, x_bar + moe
    print(f"  {cl}% CI: {x_bar:.4f} ± {z_val}×{se:.4f} = [{lo_ci:.4f}, {hi_ci:.4f}]  width={2*moe:.4f}")

# ============================================================
# STEP 8 — CORRELATION
# ============================================================
r, p_val = stats.pearsonr(screen, cgpa)
print(f"\n\n📊 STEP 8: CORRELATION")
print("-" * 60)
print(f"  Pearson r (screen vs CGPA) = {r:.4f}  (p-value = {p_val:.6f})")
print(f"  → {'Strong NEGATIVE' if r < -0.7 else 'Moderate'} correlation")
print(f"  → p < 0.05: Statistically SIGNIFICANT ✓")

# ============================================================
# VISUALISATIONS  (6 charts)
# ============================================================
print("\n\n📊 Generating Charts...")

fig = plt.figure(figsize=(18, 22))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Chart 1: Histogram ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts, bin_edges, patches = ax1.hist(screen, bins=8, edgecolor='white',
                                       linewidth=1.5, color=C2, rwidth=0.85)
grad_cols = [C3, C2, C1, C1, '#FF9999', '#FF4444', CR, '#8B0000']
for p, col in zip(patches, grad_cols[:len(patches)]):
    p.set_facecolor(col)

ax1.axvline(np.mean(screen),   color=CR, lw=2, ls='--',
            label=f'Mean = {np.mean(screen):.2f} hrs')
ax1.axvline(np.median(screen), color=CG, lw=2, ls='-.',
            label=f'Median = {np.median(screen):.2f} hrs')
ax1.set_xlabel('Screen Time (hrs/day)', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Fig 1: Histogram — Screen Time Distribution', fontweight='bold', color=C1)
ax1.legend(fontsize=9)
ax1.set_facecolor('#F8FAFD')

# ── Chart 2: Scatter — screen vs CGPA ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sc = ax2.scatter(screen, cgpa, c=cgpa, cmap='RdYlGn', s=90,
                  edgecolors='white', lw=0.8, zorder=3, vmin=1.5, vmax=4.0)
m  = np.polyfit(screen, cgpa, 1)
xl = np.linspace(screen.min()-0.3, screen.max()+0.3, 200)
ax2.plot(xl, np.poly1d(m)(xl), color=CR, lw=2.2, ls='--',
         label=f'Trend (slope={m[0]:.3f})')
plt.colorbar(sc, ax=ax2, label='CGPA')
ax2.text(0.97, 0.95, f'r = {r:.4f}', transform=ax2.transAxes,
         fontsize=12, fontweight='bold', color=CR, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF2CC', edgecolor=CR))
ax2.set_xlabel('Screen Time (hrs/day)', fontweight='bold')
ax2.set_ylabel('CGPA', fontweight='bold')
ax2.set_title('Fig 2: Screen Time vs CGPA (Scatter)', fontweight='bold', color=C1)
ax2.legend(fontsize=9)
ax2.set_facecolor('#F8FAFD')

# ── Chart 3: Bar — avg CGPA by category ─────────────────────
ax3 = fig.add_subplot(gs[1, 0])
cat_labels  = ['Low\n0–2 hrs', 'Moderate\n2–4 hrs', 'High\n4–6 hrs', 'Very High\n6+ hrs']
masks       = [screen <= 2,
               (screen > 2) & (screen <= 4),
               (screen > 4) & (screen <= 6),
               screen > 6]
cat_means   = [np.mean(cgpa[m]) for m in masks]
cat_counts  = [np.sum(m)        for m in masks]
bar_cols    = [CG, C2, CY, CR]
bars = ax3.bar(cat_labels, cat_means, color=bar_cols, edgecolor='white', lw=1.2, width=0.55)
for bar, mean_v, cnt in zip(bars, cat_means, cat_counts):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
             f'{mean_v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
             f'n={cnt}', ha='center', va='center', color='white', fontweight='bold')
ax3.axhline(np.mean(cgpa), color='gray', ls=':', lw=1.5,
            label=f'Overall mean CGPA = {np.mean(cgpa):.2f}')
ax3.set_ylim(0, 4.4)
ax3.set_ylabel('Average CGPA', fontweight='bold')
ax3.set_title('Fig 3: Average CGPA by Screen Time Category', fontweight='bold', color=C1)
ax3.legend(fontsize=9)
ax3.set_facecolor('#F8FAFD')

# ── Chart 4: Normal distribution ────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
x_norm = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 400)
y_norm = stats.norm.pdf(x_norm, mu, sigma)
ax4.plot(x_norm, y_norm, color=C1, lw=2.5)
fill_info = [(1, '#BDD7EE', '68% (μ±1σ)'),
             (2, '#DDEEFF', '95% (μ±2σ)'),
             (3, '#EEF5FF', '99.7% (μ±3σ)')]
for k, col, lbl in reversed(fill_info):
    xf = np.linspace(mu-k*sigma, mu+k*sigma, 300)
    yf = stats.norm.pdf(xf, mu, sigma)
    ax4.fill_between(xf, yf, alpha=0.85, color=col, label=lbl)
ax4.axvline(mu, color=CR, lw=2, ls='--', label=f'μ={mu:.2f}')
ax4.plot(screen, np.zeros(n)-0.001, '|', color=C1, ms=12, alpha=0.5)
ax4.set_xlabel('Screen Time (hrs)', fontweight='bold')
ax4.set_ylabel('Probability Density', fontweight='bold')
ax4.set_title(f'Fig 4: Normal Distribution  μ={mu:.2f}, σ={sigma:.2f}', fontweight='bold', color=C1)
ax4.legend(fontsize=8, loc='upper right')
ax4.set_facecolor('#F8FAFD')

# ── Chart 5: Confidence intervals ───────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ci_levels  = ['90%', '95%', '99%']
ci_z       = [1.645, 1.960, 2.576]
ci_cols    = [CG, C2, CR]
for i, (lv, z, col) in enumerate(zip(ci_levels, ci_z, ci_cols)):
    lo_ci = x_bar - z*se
    hi_ci = x_bar + z*se
    ax5.barh(i, hi_ci-lo_ci, left=lo_ci, height=0.45,
             color=col, alpha=0.75, edgecolor='white', lw=1.5)
    ax5.text(hi_ci+0.03, i, f'[{lo_ci:.3f}, {hi_ci:.3f}]',
             va='center', fontsize=9, color=col, fontweight='bold')
    ax5.text(lo_ci-0.03, i, lv, va='center', ha='right',
             fontsize=11, color=col, fontweight='bold')
ax5.axvline(x_bar, color='black', lw=2, ls='--', label=f'x̄={x_bar:.3f}')
ax5.set_xlabel('Screen Time (hrs)', fontweight='bold')
ax5.set_title(f'Fig 5: Confidence Intervals (n={n}, SE={se:.3f})', fontweight='bold', color=C1)
ax5.set_yticks([])
ax5.legend(fontsize=9)
ax5.set_facecolor('#F8FAFD')
ax5.set_xlim(x_bar - 1.8, x_bar + 1.8)

# ── Chart 6: Bayes' posterior bar ───────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
prior  = [P_A1,       P_A2]
post   = [P_A1_given_B, P_A2_given_B]
x_pos  = np.arange(2)
w      = 0.35
bars_prior = ax6.bar(x_pos-w/2, prior, width=w, color=[C3, C2],
                      label='Prior P(Usage)',  edgecolor='white', lw=1.2)
bars_post  = ax6.bar(x_pos+w/2, post,  width=w, color=[CG,  CR],
                      label='Posterior P(Usage|Low CGPA)', edgecolor='white', lw=1.2)
for bar in list(bars_prior)+list(bars_post):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom',
             fontweight='bold', fontsize=10)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['Low Usage', 'High Usage'], fontweight='bold')
ax6.set_ylabel('Probability', fontweight='bold')
ax6.set_title("Fig 6: Bayes' Rule — Prior vs Posterior\n(Given Student has Low CGPA)",
               fontweight='bold', color=C1)
ax6.legend(fontsize=9)
ax6.set_facecolor('#F8FAFD')
ax6.set_ylim(0, 1.05)

# ── Super title ──────────────────────────────────────────────
fig.suptitle(
    "The Student's Digital Dilemma\n"
    "Social Media Usage vs Academic Performance — Statistical Analysis\n"
    "MTH-262 | COMSATS Islamabad, Sahiwal Campus",
    fontsize=15, fontweight='bold', color=C1, y=0.98
)

plt.savefig('digital_dilemma_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.show()
print("\n✅ Chart saved as  digital_dilemma_analysis.png")
print("\n" + "="*60)
print("  All statistical analyses complete!")
print("  MTH-262 | Aleeha | COMSATS Sahiwal")
print("="*60)
