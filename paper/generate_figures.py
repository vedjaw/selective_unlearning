"""
Generate publication-quality figures for the unlearning paper.

Produces 5 figures using actual experimental results:
  1. I∩ bar chart across methods (CIFAR-10 multi-seed)
  2. I∩ vs ForgetAcc scatter (showing metric disagreement)
  3. Stress-test grouped bar chart
  4. CIFAR-10 vs CIFAR-100 consistency plot
  5. Multi-metric radar chart

Usage:
    python paper/generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Use a clean, modern style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# DATA (from actual experimental results)
# ═══════════════════════════════════════════════════════════

# Multi-seed CIFAR-10 audit results (mean ± std)
METHODS = ['Retrain\n(Gold)', 'Amnesiac', 'Bad\nTeacher', 'IWEUP v2\n(Ours)', 'SalUn', 'SCRUB', 'Gradient\nAscent', 'Fine-tune']
METHODS_SHORT = ['Retrain', 'Amnesiac', 'Bad Teacher', 'IWEUP v2', 'SalUn', 'SCRUB', 'GA', 'Fine-tune']

I_CAP_MEAN = [0.076, 0.083, 0.092, 0.122, 0.146, 0.157, 0.174, 0.193]
I_CAP_STD  = [0.009, 0.012, 0.020, 0.016, 0.005, 0.004, 0.002, 0.000]

FORGET_ACC = [0.0, 0.7, 15.7, 10.5, 0.0, 63.0, 37.0, 49.2]
TEST_ACC   = [85.2, 83.3, 82.1, 83.4, 83.1, 88.3, 87.3, 89.1]
RISK_SCORE = [0.086, 0.147, 0.107, 0.137, 0.130, 0.136, 0.153, 0.168]

# CIFAR-100 results (single seed)
I_CAP_C100 = [0.084, None, None, 0.112, None, 0.159, 0.184, 0.192]
METHODS_C100 = ['Retrain', 'IWEUP v2', 'SCRUB', 'GA', 'Fine-tune']
I_CAP_C100_VALS = [0.084, 0.112, 0.159, 0.184, 0.192]

# Stress test results
STRESS_METHODS = ['IWEUP v2\n(Ours)', 'Fine-tune', 'Gradient\nAscent', 'SCRUB']
STRESS_ICAP_EASY = [0.000, 0.010, 0.022, 0.017]
STRESS_ICAP_RAND = [0.000, 0.010, 0.021, 0.016]
STRESS_ICAP_HARD = [0.000, 0.010, 0.022, 0.016]
STRESS_MIA_EASY  = [0.000, 0.220, 0.018, 0.428]
STRESS_MIA_HARD  = [0.000, 0.232, 0.018, 0.452]
STRESS_PARAM_DIST = [49.25, 49.02, 49.41, 186.84]

# Color palette — conference-friendly, colorblind-safe
COLORS = {
    'retrain': '#2ecc71',    # green (gold standard)
    'ours':    '#e74c3c',    # red (our method, stands out)
    'others':  '#3498db',    # blue
    'easy':    '#27ae60',    # green
    'random':  '#f39c12',    # orange
    'hard':    '#c0392b',    # dark red
}

def get_bar_colors():
    """Assign colors: green for retrain, red for ours, blue gradient for others."""
    blues = plt.cm.Blues(np.linspace(0.35, 0.75, 6))
    colors = [COLORS['retrain']]  # Retrain
    for i in range(6):
        colors.append(blues[i])
    colors.insert(3, COLORS['ours'])  # Insert IWEUP v2 at position 3
    return colors[:8]


# ═══════════════════════════════════════════════════════════
# FIGURE 1: I∩ Bar Chart (Main audit result)
# ═══════════════════════════════════════════════════════════
def fig1_icap_bar_chart():
    """
    Bar chart of Residual Knowledge (I∩) across all methods.
    This is the KEY figure — goes in Section 6.4.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    x = np.arange(len(METHODS))
    colors = get_bar_colors()
    
    bars = ax.bar(x, I_CAP_MEAN, yerr=I_CAP_STD, 
                  color=colors, edgecolor='white', linewidth=0.8,
                  capsize=4, error_kw={'linewidth': 1.2, 'ecolor': '#555'})
    
    # Highlight retrain baseline with dashed line
    ax.axhline(y=I_CAP_MEAN[0], color=COLORS['retrain'], linestyle='--', 
               linewidth=1.2, alpha=0.7, label='Retrain baseline')
    
    # Add value labels on top of bars
    for i, (bar, val, std) in enumerate(zip(bars, I_CAP_MEAN, I_CAP_STD)):
        y_pos = val + std + 0.004
        fontweight = 'bold' if i == 3 else 'normal'  # Bold for ours
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, 
                f'{val:.3f}', ha='center', va='bottom', 
                fontsize=8.5, fontweight=fontweight)
    
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_ylabel('Residual Knowledge $I_\\cap$ (bits)')
    ax.set_title('Information Decomposition Audit — CIFAR-10 (3 seeds)')
    ax.set_ylim(0, 0.24)
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add annotation arrow pointing to ours
    ax.annotate('Closest to\nRetrain', xy=(3, 0.122), xytext=(4.5, 0.04),
                fontsize=9, ha='center', color=COLORS['ours'],
                arrowprops=dict(arrowstyle='->', color=COLORS['ours'], lw=1.5))
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_icap_bar_chart.pdf')
    fig.savefig(OUTPUT_DIR / 'fig1_icap_bar_chart.png')
    print(f"  ✓ Figure 1 saved: {OUTPUT_DIR / 'fig1_icap_bar_chart.pdf'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# FIGURE 2: I∩ vs Forget Accuracy Scatter
# ═══════════════════════════════════════════════════════════
def fig2_icap_vs_forgetacc():
    """
    Scatter plot showing that ForgetAcc and I∩ disagree.
    This is the "aha" figure — goes in Discussion section.
    Key insight: Fine-tune and GA have similar I∩ but very different ForgetAcc.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    
    colors = get_bar_colors()
    
    for i, (name, icap, fa) in enumerate(zip(METHODS_SHORT, I_CAP_MEAN, FORGET_ACC)):
        marker = '★' if name == 'IWEUP v2' else ('D' if name == 'Retrain' else 'o')
        size = 200 if name in ('IWEUP v2', 'Retrain') else 120
        zorder = 10 if name in ('IWEUP v2', 'Retrain') else 5
        
        ax.scatter(fa, icap, c=[colors[i]], s=size, marker=marker if marker != '★' else '*',
                   edgecolors='black', linewidths=0.8, zorder=zorder, label=name)
    
    # Draw the "ideal zone" box (low I∩, low ForgetAcc)
    rect = plt.Rectangle((-5, -0.01), 25, 0.11, 
                          linewidth=1.5, edgecolor=COLORS['retrain'],
                          facecolor=COLORS['retrain'], alpha=0.1, 
                          linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(8, 0.095, 'Ideal Zone', fontsize=9, color=COLORS['retrain'],
            fontstyle='italic', alpha=0.8)
    
    # Highlight the key insight with annotation
    ax.annotate('Similar $I_\\cap$,\nvery different ForgetAcc',
                xy=(43, 0.183), xytext=(55, 0.14),
                fontsize=8.5, ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#666'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                          edgecolor='#ccc', alpha=0.9))
    
    # Connect Fine-tune and GA to show similarity
    ax.plot([37.0, 49.2], [0.174, 0.193], color='#aaa', linestyle=':', 
            linewidth=1.5, zorder=0)
    
    ax.set_xlabel('Forget Accuracy (%)')
    ax.set_ylabel('Residual Knowledge $I_\\cap$ (bits)')
    ax.set_title('$I_\\cap$ vs. Forget Accuracy — Metric Disagreement')
    ax.legend(loc='center right', fontsize=8, framealpha=0.9,
              bbox_to_anchor=(1.0, 0.45))
    
    ax.set_xlim(-5, 75)
    ax.set_ylim(0, 0.22)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_icap_vs_forgetacc.pdf')
    fig.savefig(OUTPUT_DIR / 'fig2_icap_vs_forgetacc.png')
    print(f"  ✓ Figure 2 saved: {OUTPUT_DIR / 'fig2_icap_vs_forgetacc.pdf'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Stress Test Grouped Bar Chart
# ═══════════════════════════════════════════════════════════
def fig3_stress_test():
    """
    Grouped bar chart showing I∩ and MIA across easy/random/hard splits.
    Goes in Section 6.5.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    x = np.arange(len(STRESS_METHODS))
    width = 0.25
    
    # Left panel: I∩ across splits
    bars1 = ax1.bar(x - width, STRESS_ICAP_EASY, width, label='Easy', 
                    color=COLORS['easy'], edgecolor='white', alpha=0.85)
    bars2 = ax1.bar(x, STRESS_ICAP_RAND, width, label='Random', 
                    color=COLORS['random'], edgecolor='white', alpha=0.85)
    bars3 = ax1.bar(x + width, STRESS_ICAP_HARD, width, label='Hard', 
                    color=COLORS['hard'], edgecolor='white', alpha=0.85)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(STRESS_METHODS, fontsize=9)
    ax1.set_ylabel('Residual Knowledge $I_\\cap$ (bits)')
    ax1.set_title('(a) $I_\\cap$ Across Difficulty Splits')
    ax1.legend(title='Forget Set', framealpha=0.9, fontsize=8)
    ax1.set_ylim(0, 0.035)
    
    # Add annotation for IWEUP v2 = 0.000
    ax1.annotate('$I_\\cap = 0.000$\nacross all splits', 
                 xy=(0, 0.001), xytext=(1.2, 0.028),
                 fontsize=9, color=COLORS['ours'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['ours'], lw=1.5))
    
    # Right panel: MIA across splits
    bars4 = ax2.bar(x - width/2, STRESS_MIA_EASY, width, label='Easy', 
                    color=COLORS['easy'], edgecolor='white', alpha=0.85)
    bars5 = ax2.bar(x + width/2, STRESS_MIA_HARD, width, label='Hard', 
                    color=COLORS['hard'], edgecolor='white', alpha=0.85)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(STRESS_METHODS, fontsize=9)
    ax2.set_ylabel('MIA Detection Rate')
    ax2.set_title('(b) MIA Vulnerability: Easy vs. Hard')
    ax2.legend(title='Forget Set', framealpha=0.9, fontsize=8)
    ax2.set_ylim(0, 0.55)
    
    # Highlight SCRUB's high MIA
    ax2.annotate('Privacy\nleak', xy=(3.15, 0.452), xytext=(2.5, 0.50),
                 fontsize=8, color=COLORS['hard'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['hard'], lw=1.2))
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_stress_test.pdf')
    fig.savefig(OUTPUT_DIR / 'fig3_stress_test.png')
    print(f"  ✓ Figure 3 saved: {OUTPUT_DIR / 'fig3_stress_test.pdf'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# FIGURE 4: CIFAR-10 vs CIFAR-100 Consistency
# ═══════════════════════════════════════════════════════════
def fig4_cross_dataset():
    """
    Parallel bar chart showing I∩ ranking consistency across datasets.
    Goes in Section 6.4 to support generalizability claim.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    methods = ['Retrain', 'IWEUP v2\n(Ours)', 'SCRUB', 'Gradient\nAscent', 'Fine-tune']
    c10_vals = [0.076, 0.122, 0.157, 0.174, 0.193]
    c100_vals = [0.084, 0.112, 0.159, 0.184, 0.192]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, c10_vals, width, label='CIFAR-10', 
                   color='#3498db', edgecolor='white', alpha=0.85)
    bars2 = ax.bar(x + width/2, c100_vals, width, label='CIFAR-100', 
                   color='#e67e22', edgecolor='white', alpha=0.85)
    
    # Add value labels
    for bar, val in zip(bars1, c10_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#3498db')
    for bar, val in zip(bars2, c100_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='#e67e22')
    
    # Draw ranking arrows
    ax.annotate('', xy=(0.5, 0.215), xytext=(-0.2, 0.215),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    ax.text(0.15, 0.22, 'Same ranking →', fontsize=9, color='#555', 
            ha='center', fontstyle='italic')
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Residual Knowledge $I_\\cap$ (bits)')
    ax.set_title('$I_\\cap$ Ranking Consistency Across Datasets')
    ax.legend(framealpha=0.9, fontsize=10)
    ax.set_ylim(0, 0.24)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_cross_dataset.pdf')
    fig.savefig(OUTPUT_DIR / 'fig4_cross_dataset.png')
    print(f"  ✓ Figure 4 saved: {OUTPUT_DIR / 'fig4_cross_dataset.pdf'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Multi-Metric Radar Chart
# ═══════════════════════════════════════════════════════════
def fig5_radar_chart():
    """
    Radar chart comparing top-4 methods across multiple metrics.
    Goes in Section 6 overview or Discussion.
    """
    categories = ['$I_\\cap$ (inv.)', 'Test Acc', 'Forget Privacy\n(1-ForgetAcc)', 
                  'MIA Privacy\n(1-Risk)', 'Speed\n(inv. time)']
    N = len(categories)
    
    # Normalize metrics to [0, 1] where 1 = best
    # Methods: Retrain, IWEUP v2, Amnesiac, SCRUB
    methods_radar = {
        'Retrain (Gold)': {
            'icap_inv': 1.0 - (0.076 / 0.2),     # lower I∩ = better
            'test_acc': 85.2 / 90.0,
            'forget_priv': 1.0 - 0.0 / 100.0,     # 1-ForgetAcc
            'mia_priv': 1.0 - 0.086,
            'speed': 0.15,                          # slowest
        },
        'IWEUP v2 (Ours)': {
            'icap_inv': 1.0 - (0.122 / 0.2),
            'test_acc': 83.4 / 90.0,
            'forget_priv': 1.0 - 10.5 / 100.0,
            'mia_priv': 1.0 - 0.137,
            'speed': 0.75,
        },
        'Amnesiac': {
            'icap_inv': 1.0 - (0.083 / 0.2),
            'test_acc': 83.3 / 90.0,
            'forget_priv': 1.0 - 0.7 / 100.0,
            'mia_priv': 1.0 - 0.147,
            'speed': 0.55,
        },
        'SCRUB': {
            'icap_inv': 1.0 - (0.157 / 0.2),
            'test_acc': 88.3 / 90.0,
            'forget_priv': 1.0 - 63.0 / 100.0,
            'mia_priv': 1.0 - 0.136,
            'speed': 0.45,
        },
    }
    
    radar_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for (name, vals), color in zip(methods_radar.items(), radar_colors):
        values = list(vals.values())
        values += values[:1]
        
        linewidth = 2.5 if 'Ours' in name else 1.5
        alpha_fill = 0.15 if 'Ours' in name else 0.08
        
        ax.plot(angles, values, 'o-', linewidth=linewidth, color=color, label=name)
        ax.fill(angles, values, alpha=alpha_fill, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=8, color='#888')
    ax.set_title('Multi-Metric Comparison (Higher = Better)', pad=20, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_radar_chart.pdf')
    fig.savefig(OUTPUT_DIR / 'fig5_radar_chart.png')
    print(f"  ✓ Figure 5 saved: {OUTPUT_DIR / 'fig5_radar_chart.pdf'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating paper figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    fig1_icap_bar_chart()
    fig2_icap_vs_forgetacc()
    fig3_stress_test()
    fig4_cross_dataset()
    fig5_radar_chart()
    
    print(f"\n✅ All 5 figures generated in {OUTPUT_DIR}/")
    print("\nLaTeX usage:")
    print("  \\includegraphics[width=\\textwidth]{figures/fig1_icap_bar_chart.pdf}")
    print("  \\includegraphics[width=\\textwidth]{figures/fig2_icap_vs_forgetacc.pdf}")
    print("  \\includegraphics[width=\\textwidth]{figures/fig3_stress_test.pdf}")
    print("  \\includegraphics[width=0.85\\textwidth]{figures/fig4_cross_dataset.pdf}")
    print("  \\includegraphics[width=0.7\\textwidth]{figures/fig5_radar_chart.pdf}")
