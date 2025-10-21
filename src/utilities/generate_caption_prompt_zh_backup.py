"""
Backup of the current Chinese prompt templates used by generate_caption.py
"""

# =============================================================================
# CHINESE PROMPT TEMPLATES (BACKUP)
# =============================================================================

SYSTEM_PROMPT_ZH = """你是一名音频分析与工业声学领域的专家。你的任务是基于给定的工业机器声音的频谱分析特征，生成一段自然语言描述（Caption），准确刻画该声音的客观声学特性。

请侧重于声学与物理层面的可测量特征，避免主观感受性表述。生成的描述应有助于在异常检测场景下，将该声音与其他工业机器声音区分开来。"""

CAPTION_PROMPT_TEMPLATE_ZH = """以下为某工业机器音频信号提取的特征：

## 频谱相关特征（Spectral）
- 频谱质心方差（Spectral Centroid Var）: {spectral_centroid_var:.6f}
- 频谱流方差（Spectral Flux Var）: {spectral_flux_var:.6f}
- 频谱扩展均值（Spectral Spread Mean）: {spectral_spread_mean:.6f}
- 频谱扩展标准差（Spectral Spread Std）: {spectral_spread_std:.6f}
- 频谱滚降均值（Spectral Rolloff Mean）: {spectral_rolloff_mean:.6f}
- 频谱滚降标准差（Spectral Rolloff Std）: {spectral_rolloff_std:.6f}
- 频谱平坦度均值（Spectral Flatness Mean）: {spectral_flatness_mean:.6f}
- 频谱平坦度标准差（Spectral Flatness Std）: {spectral_flatness_std:.6f}

## 能量与幅度特征（Energy/Amplitude）
- 能量包络方差（Energy Envelope Var）: {energy_envelope_var:.6f}
- 峰-峰值标准差（Peak-to-Peak Std）: {peak_to_peak_std:.6f}
- 峰-峰值（Peak-to-Peak Value）: {peak_to_peak_value:.6f}
- 峰度因子（Crest Factor）: {crest_factor:.6f}
- 峰值幅度（Peak Amplitude）: {peak_amplitude:.6f}

## 时域特征（Temporal）
- 全局过零率（Zero Crossing Rate Global）: {zero_crossing_rate_global:.6f}
- 帧级过零率（Zero Crossing Rate Frame）: {zero_crossing_rate_frame:.6f}
- 复合方差（Composite Var）: {composite_var:.6f}

## MFCC 与统计特征（MFCC/Statistics）
- MFCC 方差（MFCC Var）: {mfcc_var:.6f}
- 偏度（Skewness）: {skewness:.6f}
- 峰度（Kurtosis）: {kurtosis:.6f}

## 包络特性（Envelope）
- 起音时间（Envelope Attack Time）: {envelope_attack_time:.6f}
- 衰减时间（Envelope Decay Time）: {envelope_decay_time:.6f}
- 包络峰度（Envelope Kurtosis）: {envelope_kurtosis:.6f}
- 包络偏度（Envelope Skewness）: {envelope_skewness:.6f}
- 变异系数（Env. Coefficient of Variation）: {envelope_coefficient_of_variation:.6f}
- 起音强度（Envelope Attack Strength）: {envelope_attack_strength:.6f}
- 维持比（Envelope Sustain Ratio）: {envelope_sustain_ratio:.6f}
- 平滑度（Envelope Smoothness）: {envelope_smoothness:.6f}

## 心理声学特征（Psychoacoustic）
- 总谐波失真 THD: {thd:.6f}
- 粗糙度（Roughness）: {roughness:.6f}
- 锐度（Sharpness）: {sharpness:.6f}
- 起伏强度（Fluctuation Strength）: {fluctuation_strength:.6f}

## 频段能量分布（0-8000 Hz）
- 0-1000 Hz: Mean={band_0_1000_energy_ratio_mean:.6f}, Std={band_0_1000_energy_ratio_std:.6f}
- 1000-2000 Hz: Mean={band_1000_2000_energy_ratio_mean:.6f}, Std={band_1000_2000_energy_ratio_std:.6f}
- 2000-3000 Hz: Mean={band_2000_3000_energy_ratio_mean:.6f}, Std={band_2000_3000_energy_ratio_std:.6f}
- 3000-4000 Hz: Mean={band_3000_4000_energy_ratio_mean:.6f}, Std={band_3000_4000_energy_ratio_std:.6f}
- 4000-5000 Hz: Mean={band_4000_5000_energy_ratio_mean:.6f}, Std={band_4000_5000_energy_ratio_std:.6f}
- 5000-6000 Hz: Mean={band_5000_6000_energy_ratio_mean:.6f}, Std={band_5000_6000_energy_ratio_std:.6f}
- 6000-7000 Hz: Mean={band_6000_7000_energy_ratio_mean:.6f}, Std={band_6000_7000_energy_ratio_std:.6f}
- 7000-8000 Hz: Mean={band_7000_8000_energy_ratio_mean:.6f}, Std={band_7000_8000_energy_ratio_std:.6f}

基于以上所有特征，请生成一段完整的技术性说明文字（3-5句），需覆盖：
1. 整体音色与频谱特性（质心、扩展、滚降、平坦度）
2. 时间动态与包络特性（起音、衰减、稳定性、平滑度）
3. 主导频率范围与各频段能量分布
4. 幅度与能量特性（峰值、峰度因子、方差等）
5. 统计属性（偏度、峰度、MFCC 模式）
6. 感知品质（粗糙度、锐度、THD 所反映的谐波性、起伏强度）

请确保描述覆盖所有特征类别，仅输出描述文本，不要输出其他内容。"""
