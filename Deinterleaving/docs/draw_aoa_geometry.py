"""画清楚 AoA 正负 141 度的几何位置。示例约定：右方为 0 度，逆时针为正。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6.3), dpi=180)
fig.patch.set_facecolor("white")
ax.set_aspect("equal")
ax.set_xlim(-4.7, 5.1)
ax.set_ylim(-3.7, 3.7)
ax.axis("off")

# 接收机在原点；方向线严格按极角计算。
ax.add_patch(Circle((0, 0), .19, facecolor="#25374a", edgecolor="none", zorder=5))
ax.text(0, -.43, "接收机", ha="center", va="top", fontsize=14, weight="bold")
ax.annotate("", xy=(4.4, 0), xytext=(.25, 0),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.6))
ax.text(3.3, .2, "0° 参考方向", fontsize=12, color="#444")

for angle, name, color, yoff in ((141, "雷达 A", "#1765b1", .25), (-141, "雷达 B", "#d87418", -.27)):
    theta = np.deg2rad(angle)
    p = 3.9 * np.array([np.cos(theta), np.sin(theta)])
    ax.plot([0, p[0]], [0, p[1]], color=color, lw=2.3, ls="--")
    ax.scatter([p[0]], [p[1]], s=170, marker="^", color=color, zorder=6)
    ax.text(p[0]-.05, p[1]+yoff, name, ha="center", fontsize=13, color=color, weight="bold")
    # 发射的电波沿视线传向接收机。
    outer=.68*p; inner=.41*p
    ax.add_patch(FancyArrowPatch(outer, inner, arrowstyle="-|>", mutation_scale=17,
                                 color=color, lw=2.2, zorder=7))

# 从 0° 到两条射线的完整弧；较短夹角跨越左方 180°。
ax.add_patch(Arc((0,0), 2.15, 2.15, theta1=0, theta2=141, color="#1765b1", lw=1.7))
ax.add_patch(Arc((0,0), 2.72, 2.72, theta1=219, theta2=360, color="#d87418", lw=1.7))
ax.add_patch(Arc((0,0), 1.20, 1.20, theta1=141, theta2=219, color="#9a406e", lw=2.5))
ax.text(-1.7, 1.17, "+141°", color="#1765b1", fontsize=13, weight="bold")
ax.text(-2.03, -1.44, "−141°", color="#d87418", fontsize=13, weight="bold")
ax.text(-1.40, -.02, "78°", color="#9a406e", fontsize=14, weight="bold")
ax.text(-3.75, 0, "两方向的最小夹角", ha="center", va="center", color="#9a406e", fontsize=12)
ax.text(0, 3.35, "AoA 方向示意", ha="center", fontsize=18, weight="bold")
ax.text(0, -3.25, "示例约定：接收机右方为 0°，逆时针为正；真实坐标须以接收机标定为准。",
        ha="center", fontsize=10.5, color="#555")

path = Path(__file__).with_name("AOA正负141度示意图.png")
fig.savefig(path, bbox_inches="tight", facecolor="white")
print(path)
