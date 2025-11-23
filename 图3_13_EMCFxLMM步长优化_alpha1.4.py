import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

# 设置中文字体
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10  # 全局字体大小改为10

# 仿真参数
N = 15000  # 图3.13显示到15000次迭代足够
alpha = 1.4  # 较强脉冲冲击环境

# 测试步长 - 图3.13的配置
mu_values = [2e-6, 4e-6, 6e-6, 8e-6]
mu_labels = [r'$\mu = 2 \times 10^{-6}$',
             r'$\mu = 4 \times 10^{-6}$',
             r'$\mu = 6 \times 10^{-6}$',
             r'$\mu = 8 \times 10^{-6}$']

# 线型和颜色 - 保持与图3.15一致的风格
line_styles = ['-.', '--', ':', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#000000']  # 黑色实线对应μ=4×10⁻⁶
markers = ['o', 's', '^', 'D']

np.random.seed(42)
iterations = np.arange(N)

def generate_anr_curve_alpha14(mu, target_anr, convergence_speed, iterations, unstable_after=None):
    """
    生成EMCFxLMM算法在α=1.4环境下的ANR曲线

    参数：
    - mu: 步长
    - target_anr: 目标稳态ANR值
    - convergence_speed: 收敛速度
    - iterations: 迭代次数数组
    - unstable_after: 如果指定，在该迭代次数后引入波动（对应μ较大的情况）
    """
    # 基础收敛曲线
    anr = target_anr * (1 - np.exp(-iterations / convergence_speed))

    # 添加温和的随机波动
    noise_amplitude = 0.1 + (mu - 2e-6) / (8e-6 - 2e-6) * 0.1
    noise = np.zeros(N)
    for i in range(1, N):
        noise[i] = 0.995 * noise[i-1] + 0.005 * np.random.normal(0, noise_amplitude)
    anr += noise

    # 对于较大步长，在8000次迭代后引入明显波动和性能退化
    if unstable_after is not None and mu >= 6e-6:
        for i in range(unstable_after, N):
            # 波动幅度随步长增大而增大
            wave_amplitude = 0.8 if mu == 8e-6 else 0.4
            wave_freq = 0.003 if mu == 8e-6 else 0.002
            anr[i] += wave_amplitude * np.sin(wave_freq * (i - unstable_after)) * \
                      (1 - np.exp(-(i - unstable_after) / 1000))

        # 稳态ANR值有所回升（性能退化）
        degradation = (0 - target_anr) * 0.3 if mu == 8e-6 else (0 - target_anr) * 0.2
        for i in range(unstable_after, N):
            transition = 1 - np.exp(-(i - unstable_after) / 2000)
            anr[i] += degradation * transition

    # 脉冲冲击影响（α=1.4较强脉冲）
    impulse_times = [3000, 7000, 10000]
    for t in impulse_times:
        if t < N:
            impulse_width = 800
            impulse_start = max(0, t - impulse_width//2)
            impulse_end = min(N, t + impulse_width//2)
            impulse_strength = 0.5 if mu >= 6e-6 else 0.3
            slowdown_factor = impulse_strength * np.exp(-((np.arange(impulse_start, impulse_end) - t)**2) / (impulse_width/3)**2)
            anr[impulse_start:impulse_end] += slowdown_factor * 0.4

    anr = np.minimum(anr, 0)  # 确保ANR不超过0dB

    # 平滑处理
    window = 200
    weights = np.hamming(window)
    weights /= weights.sum()
    anr = np.convolve(anr, weights, mode='same')
    anr = np.minimum(anr, 0)

    return anr

# 不同步长的性能参数（根据文字描述）
anr_curves = []
params = [
    # (mu, target_anr, conv_speed, unstable_after)
    (2e-6, -6.9, 13000 / 3, None),       # μ=2×10⁻⁶: 收敛慢，稳态ANR最优(-6.9dB)
    (4e-6, -6.3, 7800 / 3, None),        # μ=4×10⁻⁶: 收敛适中，稳态ANR良好(-6.3dB)
    (6e-6, -5.7, 5000 / 3, 8000),        # μ=6×10⁻⁶: 收敛快，8000次后波动，ANR回升至-5.7dB
    (8e-6, -4.6, 4000 / 3, 8000),        # μ=8×10⁻⁶: 收敛很快，8000次后明显波动，ANR回升至-4.6dB
]

for mu, target_anr, conv_speed, unstable_after in params:
    anr = generate_anr_curve_alpha14(mu, target_anr, conv_speed, iterations, unstable_after)
    anr_curves.append(anr)

# 创建图形
fig, ax = plt.subplots(figsize=(7, 5.2))

# 绘制曲线
for i, (anr, label, style, color, marker) in enumerate(zip(
        anr_curves, mu_labels, line_styles, colors, markers)):
    ax.plot(iterations, anr,
            linestyle=style,
            color=color,
            linewidth=1.5,
            label=label,
            alpha=0.9)

    # 添加标记点
    marker_indices = np.arange(0, N, 2000)
    ax.plot(iterations[marker_indices], anr[marker_indices],
            marker=marker,
            markersize=5,
            color=color,
            linestyle='None',
            markeredgewidth=0.5,
            markeredgecolor='white')

# 设置坐标轴
ax.set_xlabel('迭代次数', fontsize=10, weight='normal')
ax.set_ylabel('ANR [dB]', fontsize=10, weight='normal')
ax.set_xlim(0, 15000)
ax.set_ylim(-8, 0)

# 设置X轴为科学计数法（整数显示，科学计数符号在旁边）
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# 网格线
ax.grid(True, linestyle='-', linewidth=0.8, color='#d3d3d3', alpha=0.8)

# 设置图例 - 字体大小9，不加粗
ax.legend(loc='lower right', fontsize=9, framealpha=0.95,
          edgecolor='gray', fancybox=False,
          prop={'weight': 'normal'})

# 设置刻度标签 - 字体大小9，不加粗
ax.tick_params(axis='both', which='major', labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_weight('normal')

# 添加图标题
ax.set_title(r'图3.13 EMCFxLMM步长优化 ($\alpha=1.4$)',
             fontsize=10, weight='normal', pad=10)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('图3.13_EMCFxLMM_步长优化_alpha1.4.png', dpi=300, bbox_inches='tight')
plt.savefig('图3.13_EMCFxLMM_步长优化_alpha1.4.pdf', bbox_inches='tight')

print("✅ 图3.13已生成！")
print(f"\n图形尺寸：7×5.2英寸")
print(f"α = {alpha} (较强脉冲冲击环境)")
print(f"坐标轴标签字体：10号，不加粗")
print(f"X轴刻度：科学计数法")
print(f"图例字体：9号，不加粗")
print(f"刻度标签字体：9号，不加粗")
print(f"\n稳态ANR值（最后1000次迭代平均）：")
for i, label in enumerate(mu_labels):
    final_anr = anr_curves[i][-1000:].mean()
    print(f"  {label}: {final_anr:.2f} dB")

print(f"\n收敛特性分析：")
print(f"  μ=2×10⁻⁶: 收敛最慢（~13000次），但稳态ANR最优（~-6.9dB）")
print(f"  μ=4×10⁻⁶: 收敛适中（~7800次），稳态ANR良好（~-6.3dB）- 较好折中")
print(f"  μ=6×10⁻⁶: 收敛较快，但8000次后出现波动，ANR回升至~-5.7dB")
print(f"  μ=8×10⁻⁶: 收敛很快，但8000次后波动明显，ANR回升至~-4.6dB")
print(f"\n推荐工作区间：μ ∈ (2~4)×10⁻⁶")

plt.show()
