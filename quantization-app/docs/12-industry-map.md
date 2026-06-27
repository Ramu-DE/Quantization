# Who Built What in Quantization — Industry Map

## Why Quantization Exists

The core problem: Modern AI models are too large to run efficiently.

| Model | FP32 Size | GPU RAM Needed | Consumer GPU |
|-------|-----------|----------------|--------------|
| LLaMA-2-7B | 28 GB | ~30 GB | Won't fit on RTX 4090 (24GB) |
| LLaMA-2-70B | 280 GB | ~290 GB | Needs 4x A100 80GB |
| GPT-4 class (1.8T) | 7.2 TB | Impossible | Entire data center |

With INT4 quantization:

| Model | Quantized Size | Now Fits On |
|-------|---------------|-------------|
| LLaMA-2-7B | 3.9 GB | Any laptop with 8GB RAM |
| LLaMA-2-70B | 35 GB | Single A100 or 2x RTX 4090 |

---

## NVIDIA — Making GPUs Do More with Less

| What | Purpose | Used By |
|------|---------|---------|
| **TensorRT** | INT8/FP8 inference engine | Every company deploying on NVIDIA GPUs |
| **ModelOpt** | PTQ + QAT toolkit (FP8, INT4, NVFP4) | Enterprise AI deployments |
| **FP8 (Hopper H100)** | New 8-bit float format in hardware | Cloud providers (AWS, Azure, GCP) |

**Why:** NVIDIA sells GPUs. Faster inference = more customers buy GPUs. They built quantization INTO the silicon (Tensor Cores).

---

## Microsoft — Running AI at Azure Scale

| What | Purpose | Used By |
|------|---------|---------|
| **ONNX Runtime** | Cross-platform quantized inference | Windows, Xbox, Azure, Office |
| **DeepSpeed/ZeroQuant** | Quantize GPT-3 scale models without retraining | Azure OpenAI Service |
| **Microscaling (MX)** | Next-gen format (co-developed with AMD, Intel, Meta, NVIDIA, Qualcomm) | Future hardware (2025+) |

**Why:** Microsoft runs Azure + OpenAI. Every 1% efficiency = millions saved in GPU costs for ChatGPT/Copilot.

---

## Google — From Phones to TPUs

| What | Purpose | Used By |
|------|---------|---------|
| **TensorFlow Lite QAT** | Quantize models for Android/Edge | Every Android phone running ML |
| **gemmlowp** | Low-precision matrix multiply library | Android Neural Networks API |
| **Gemini quantization** | Serve Gemini models efficiently | Google Search, Bard, Pixel |
| **TPU bfloat16** | Invented BFloat16 format | Google TPU v2+ for training |

**Why:** Google runs ML on billions of phones (Android) and in data centers (Search, YouTube). BFloat16 was literally invented at Google Brain.

---

## Meta (Facebook) — Open Source LLMs

| What | Purpose | Used By |
|------|---------|---------|
| **PyTorch Quantization** | Built-in PTQ/QAT framework | Most AI researchers worldwide |
| **LLaMA quantized models** | Released models that community quantizes | Everyone running local LLMs |
| **FBGEMM** | Quantized matrix multiply for x86 | Facebook's production inference |

**Why:** Meta serves 3B+ users. Every feed recommendation, every Reel, every ad uses ML. Also open-sourced LLaMA which drove the local LLM revolution.

---

## Academic / Open Source — The LLM Revolution

| Who | What | Solved |
|-----|------|--------|
| **Frantar et al. (IST Austria)** | **GPTQ** | First to quantize 175B models to 3-4 bit in hours |
| **MIT HAN Lab (Song Han)** | **AWQ** | Faster than GPTQ, better generalization, 1% salient weights insight |
| **MIT HAN Lab** | **SmoothQuant** | Solved activation outlier problem, enables W8A8 for LLMs |
| **Georgi Gerganov** | **GGML/GGUF (llama.cpp)** | Run LLMs on CPU/laptop — no GPU needed |
| **Shanghai AI Lab** | **OmniQuant** | Learnable clipping + transformation, PTQ-efficient |
| **Tim Dettmers (UW)** | **bitsandbytes / QLoRA** | 4-bit fine-tuning on single GPU, democratized LLM training |
| **Qualcomm AI Research** | **White Paper on Quantization** | Definitive engineering guide used industry-wide |
| **Turboderp** | **EXL2 (ExLlamaV2)** | Variable bit-width per layer, optimized for consumer GPUs |
| **AQLM team** | **AQLM** | 2-bit quantization using vector codebooks |
| **Mobius Labs** | **HQQ** | PTQ without needing calibration data at all |

---

## Intel / AMD — CPU and Competitor Silicon

| Who | What | Purpose |
|-----|------|---------|
| **Intel** | Neural Compressor, VNNI instructions | Quantized inference on CPUs (no GPU needed) |
| **AMD** | ROCm quantization, MI300X FP8 | Compete with NVIDIA on AI inference |
| **AMD + Microsoft + Intel + Meta + NVIDIA + Qualcomm** | **MX Formats** | Industry standard for next-gen quantization |

---

## Do You Actually Need Quantization?

### Yes, if:
- You want to run LLMs locally (not pay cloud API costs)
- You're deploying to edge/mobile devices
- You need faster inference (2-4x speed)
- You're serving millions of requests (cost reduction)
- Your model doesn't fit in available GPU memory

### No, if:
- Your model is small (< 500M params) and you have adequate hardware
- You're still training (keep FP32/BF16 for training)
- Accuracy is absolutely critical (medical/legal) and you can afford the compute
- You're using an API service (they handle it on their end)

---

## The Real-World Trade-off

```
FP32:  Maximum accuracy, 4x memory, 1x speed
INT8:  ~0.5% accuracy loss, 1x memory, 2-4x speed
INT4:  ~1-2% accuracy loss, 0.5x memory, 3-6x speed
```

For 99% of inference use cases, the accuracy loss is imperceptible but the cost/speed improvement is massive. That's why every production deployment (ChatGPT, Claude, Gemini) uses quantization internally.

---

## The Business Bottom Line

| Company | Annual AI Compute Cost | Quantization Savings |
|---------|----------------------|---------------------|
| Microsoft (OpenAI/Azure) | ~$10B+ | 4x quantization = saves ~$7B |
| Google (Gemini/Search) | ~$8B+ | BFloat16 + INT8 on TPUs = saves ~$4B |
| Meta (Recommendations) | ~$5B+ | INT8 inference = serves 3B users affordably |
| Startups/Individuals | $0-$100/mo | GGUF lets you run 70B models on a $1000 laptop |

**The punchline:** Quantization isn't optional — it's what makes AI economically viable. Without it, ChatGPT would cost $200/month per user instead of $20.

---

## Timeline of Key Developments

| Year | Milestone |
|------|-----------|
| 2015 | Google introduces quantization in TensorFlow for mobile |
| 2018 | Jacob et al. publish foundational quantization paper |
| 2019 | NVIDIA adds INT8 Tensor Cores (Turing architecture) |
| 2019 | Google Brain invents BFloat16 for TPU training |
| 2020 | NVIDIA TensorRT 7 with automatic INT8 calibration |
| 2021 | Qualcomm publishes definitive quantization white paper |
| 2022 | GPTQ — first practical 3-4 bit quantization for 175B models |
| 2022 | NVIDIA Hopper (H100) introduces FP8 hardware |
| 2022 | SmoothQuant solves activation outlier problem |
| 2023 | AWQ — activation-aware weight quantization (MIT) |
| 2023 | llama.cpp/GGUF — runs LLMs on laptops |
| 2023 | QLoRA — 4-bit fine-tuning on consumer GPUs |
| 2023 | OmniQuant — learnable quantization parameters |
| 2023 | MX Formats — industry consortium for next-gen formats |
| 2024 | HQQ — quantization without calibration data |
| 2024 | NVIDIA Blackwell (B200) — FP4 hardware support |
| 2025 | MX formats appearing in AMD MI350+ silicon |
