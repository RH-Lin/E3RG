# E3RG
** [MM 2025 Grand Challenge] ** Official Implementation for "E3RG: Building Explicit Emotion-driven Empathetic Response Generation System with Multimodal Large Language Model"

[![arXiv](https://img.shields.io/badge/arXiv-2508.12854-red)](https://arxiv.org/abs/2508.12854)

### 🥉Top-1 Solution for the $2^{nd}$ task in The ACM Multimedia 2025 Grand Challenge of Avatar-based Multimodal Empathetic Conversation

Related Website: 

[https://avamerg.github.io/MM25-challenge](https://avamerg.github.io/MM25-challenge)

[https://github.com/AvaMERG/AvaMERG-Pipeline](https://github.com/AvaMERG/AvaMERG-Pipeline)

🔥 This codebase is a demo version used to generate responded talking speeches or videos with multimodal large language model.

## Reproduce Steps

### 🛠️ 1. Installation:

```bash
pip install -r requirements.txt
```

Environment for different third-parties codebases pls refer to their official website: 

[Ola-Omni](https://github.com/Ola-Omni/Ola) as MLLM for response (can be replaced into any LLM)

[DICE-Talk](https://github.com/toto222/DICE-Talk) for talking-head generation

[OpenVoice](https://github.com/myshell-ai/OpenVoice/blob/main/docs/USAGE.md#openvoice-v2) for text-to-speech generation

[MeloTTS](https://github.com/myshell-ai/MeloTTS) provided as Base Speakers for tts

Our codebase builds upon above codebases. We extend our gratitude to their open-source models and excellent works, which have enabled us to further our exploration. 

### 🚀 2. Demo Inference:

Output both tts speech and talking-head videos:

```bash
sh infer_all.sh
```

Output both tts speech only:

```bash
sh infer_tts_openvoice.sh
```

### 📈 3. Quantitative Experiment Result

| LLM/MLLM Model            | HIT  | Dist-1 | Dist-2 |
|----------------------------|------|--------|--------|
| **Text-only LLM**          |      |        |        |
| Vicuna-1.5-7B [6]          | 46.0 | 0.825  | 0.960  |
| Llama-3-8B [13]            | 59.4 | 0.849  | 0.985  |
| InternLM3-8B [2]           | 65.3 | 0.943  | 0.997  |
| Qwen2.5-7B [38]            | 69.3 | 0.967  | 0.997  |
| Qwen2.5-7B (1-shot)        | 70.7 | 0.977  | 0.999  |
| Qwen2.5-7B (3-shot)        | 73.2 | 0.978  | 0.999  |
| MiniCPM4-8B [48]           | 73.9 | 0.983  | 0.999  |
| MiniCPM4-8B (1-shot)       | **74.7** | 0.984  | 0.999  |
| **MiniCPM4-8B (3-shot)**       | 74.2 | **0.985** | 0.999  |
|----------------------------|------|--------|--------|
| **Omni-Modal LLM**         |      |        |        |
| MiniCPM-o 2.6 8B [60]      | 65.8 | 0.952  | 0.996  |
| Qwen2.5-Omni-7B [56]       | 72.3 | 0.986  | 0.997  |
| Ola-Omni-7B [29]           | 75.6 | 0.986  | 0.999  |
| Ola-Omni-7B (1-shot)       | 76.1 | 0.989  | 0.999  |
| **Ola-Omni-7B (3-shot)**       | **76.3** | **0.990** | 0.999  |

### 💡 4. Human Evaluation

| Team      | Emotional Expressiveness | Multimodal Consistency | Naturalness | Average |
|-----------|---------------------------|-------------------------|-------------|---------|
| It’s MyGO | 3.5                       | 3.5                     | 3.2         | 3.40    |
| AI4AI     | 3.6                       | 3.8                     | **4.1**     | 3.83    |
| **Ours**  | **4.3**                   | **4.0**                 | 3.8         | **4.03**|

