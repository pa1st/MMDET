"""
-*- coding: utf-8 -*-
@Time    : 2025/5/6 19:30
@Author  : PA1ST
@Software: PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 顶刊风格设置
sns.set_theme(style="whitegrid", font_scale=1.3)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 1.2,
    "grid.color": "#EEEEEE",
    "grid.linestyle": "--",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.labelcolor": "#222222",
    "axes.titlesize": 14,
    "legend.frameon": False,
    "legend.fontsize": 12,
    "figure.dpi": 200
})

# ====== 模拟输入数据 ======
categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
group1_values = [0.72, 0.85, 0.65, 0.90]
group2_values = [0.60, 0.88, 0.70, 0.80]
group_labels = ['Method A', 'Method B']

# ====== 柱状图绘制 ======
x = np.arange(len(categories))
width = 0.35  # 柱子宽度

# 自定义配色（低饱和但有对比）
colors = ['#4477AA', '#CC6677']

fig, ax = plt.subplots(figsize=(12, 5))

bar1 = ax.bar(x - width / 2, group1_values, width, label=group_labels[0], color=colors[0])
bar2 = ax.bar(x + width / 2, group2_values, width, label=group_labels[1], color=colors[1])
# 设置 Y 轴范围防止标签溢出
max_height = max(max(group1_values), max(group2_values))
plt.ylim(0, max_height * 1.1)

# 轴标签与标题
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Category', fontsize=12)
ax.set_title('Performance Comparison by Category', fontsize=14, weight='semibold')
ax.set_xticks(x)
ax.set_xticklabels(categories)

# 图例设置
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# 柱顶显示数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


add_labels(bar1)
add_labels(bar2)

plt.tight_layout()
plt.show()

# 三柱对比 → 加 group3_values + 一组颜色 + 多余 legend
#
# 水平柱状图 → 改 barh 并转置输入
#
# 导出 PDF → 用 plt.savefig('fig.pdf', dpi=300, bbox_inches='tight')
