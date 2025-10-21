"""
Audio Caption Generator using LLM
Generates natural language captions from industrial audio spectral analysis features
"""

import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
import os
from dotenv import load_dotenv


# =============================================================================
# PROMPT TEMPLATE - Modify this to change caption generation behavior
# =============================================================================

SYSTEM_PROMPT = """你是一名音频分析与工业声学领域的专家。你的任务是基于给定的工业机器声音的频谱分析特征，生成一段自然语言描述（Caption），准确刻画该声音的客观声学特性。

请侧重于声学与物理层面的可测量特征，避免主观感受性表述。生成的描述应有助于在异常检测场景下，将该声音与其他工业机器声音区分开来。"""

CAPTION_PROMPT_TEMPLATE = """以下为某工业机器音频信号提取的特征：

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
7. 能量分布特征（各频段能量占比、能量包络变化、能量集中度等）

请确保描述覆盖所有特征类别，特别关注能量相关的详细描述，每个频段都应该描述到，仅输出描述文本，不要输出其他内容。"""

# =============================================================================
# End of Prompt Template
# =============================================================================


class AudioCaptionGenerator:
    """Generate audio captions from spectral analysis features using OpenAI-compatible API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
        prompt_template: str = CAPTION_PROMPT_TEMPLATE
    ):
        """
        Initialize the caption generator

        Args:
            api_key: API key (reads from .env -> OPENAI_API_KEY or DOUABAO_API_KEY if None)
            base_url: Base URL (reads from .env -> OPENAI_BASE_URL/DOUABAO_BASE_URL or defaults to Doubao)
            system_prompt: System prompt for the LLM
            prompt_template: Template for formatting features into prompt
        """
        # Load .env once at initialization time
        load_dotenv()

        # Resolve API key from explicit arg, then common env var names
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DOUABAO_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")

        # Resolve base URL priority: explicit arg -> env -> Doubao default
        env_base_url = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("DOUABAO_BASE_URL")
            or "https://doubao.zwchat.cn/v1"
        )
        resolved_base_url = base_url or env_base_url

        # Initialize OpenAI client (supports OpenAI-compatible APIs)
        client_kwargs = {"api_key": self.api_key, "base_url": resolved_base_url}

        self.client = OpenAI(**client_kwargs)
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    def create_prompt(self, features: Dict[str, float]) -> str:
        """
        Create a structured prompt for the LLM to generate audio caption

        Args:
            features: Dictionary containing audio feature names and values

        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(**features)

    def generate_caption(
        self,
        features: Dict[str, float],
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a caption for the given audio features

        Args:
            features: Dictionary of audio features
            model: Model to use (default: gpt-4o, can use gpt-4, gpt-3.5-turbo, etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)

        Returns:
            Generated caption string
        """
        user_prompt = self.create_prompt(features)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Be robust to providers returning None content
        try:
            choice0 = response.choices[0]
            # Some SDKs expose .message.content, others may use dict-like access
            message = getattr(choice0, "message", None) or getattr(choice0, "delta", None) or {}
            content = None
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if content is None:
                # Fallback to text field if present
                content = getattr(choice0, "text", None)
            if content is None:
                return ""
            return str(content).strip()
        except Exception:
            return ""

    def generate_batch_captions(
        self,
        features_list: list[Dict[str, float]],
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Generate captions for multiple audio samples

        Args:
            features_list: List of feature dictionaries
            model: Model to use
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of generated captions
        """
        captions = []
        for i, features in enumerate(features_list):
            print(f"Generating caption {i+1}/{len(features_list)}...")
            caption = self.generate_caption(features, model, max_tokens, temperature)
            captions.append(caption)

        return captions


def load_example_features(filepath: str = "example_features.json") -> Dict[str, Dict[str, float]]:
    """Load example features from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_model_list(filepath: str = "model_list.txt") -> List[str]:
    """Load newline-separated model names from a text file"""
    if not os.path.exists(filepath):
        return []
    models: List[str] = []
    with open(filepath, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                models.append(name)
    return models

def main():
    """Example usage"""

    # Load example features from file
    examples = load_example_features()
    example_features = examples["example_1"]

    # Initialize generator (reads API key and base URL from .env by default)
    generator = AudioCaptionGenerator()

    # Try model list; if present, iterate over models; else, single default run
    models = load_model_list()
    if models:
        print(f"Found {len(models)} models in model_list.txt. Generating captions for each...")
        for model in models:
            print(f"\nModel: {model}")
            print("Generating audio caption...")
            caption = generator.generate_caption(example_features, model=model)

            print("=" * 80)
            print(f"GENERATED CAPTION ({model}):")
            print("=" * 80)
            print(caption)
            print("=" * 80)

            # Save per-model output
            safe_model = model.replace('/', '_')
            output_file = f"audio_caption_example_{safe_model}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Model: {model}\n\n")
                f.write(f"Features:\n{json.dumps(example_features, indent=2)}\n\n")
                f.write(f"Caption:\n{caption}\n")

            print(f"Caption saved to {output_file}")
    else:
        print("Generating audio caption...")
        print(f"Using {len(example_features)} features\n")

        caption = generator.generate_caption(example_features)

        print("=" * 80)
        print("GENERATED CAPTION:")
        print("=" * 80)
        print(caption)
        print("=" * 80)

        # Save to file
        output_file = "audio_caption_example.txt"
        with open(output_file, 'w') as f:
            f.write(f"Features:\n{json.dumps(example_features, indent=2)}\n\n")
            f.write(f"Caption:\n{caption}\n")

        print(f"\nCaption saved to {output_file}")


if __name__ == "__main__":
    main()
