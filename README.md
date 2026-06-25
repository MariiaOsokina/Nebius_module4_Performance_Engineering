# Nebius_module4_Performance_Engineering
AI Performance Engineering | London


## Homework3: Advanced LLM Acceleration: Speculative Decoding & Quantization
Target hardware: 1x NVIDIA H100 80GB
Base model: Qwen/Qwen3-8B

### 1. Why LLM inference is slow — the core bottleneck
Memory bandwidth, not compute, limits decoding. Each new token streams all 8B weights from GPU memory; the H100's compute sits mostly idle waiting. Almost every optimisation here attacks that one bottleneck.
The latency metrics: TTFT (time to first token), TPOT (time per output token after the first), throughput (tokens/sec). You learned to read which one a given optimisation moves.

### 2. Speculative decoding (EAGLE-3)
The trick: a small, cheap draft model guesses several tokens ahead; the big verifier checks them all in one forward pass. Accepted guesses are "free" tokens — no separate weight-streaming pass.
Why it works below 100% acceptance: even ~1.4 accepted tokens per verifier pass raises throughput, because decoding is bandwidth-bound.
EAGLE-3 specifically: the draft is a tiny one-layer "head" that reuses the verifier's internal hidden states (layers 2/18/33 + last), so it predicts well for its size.
Acceptance decays with position: pos-0 ~36%, pos-1 ~9%, pos-2 ~2% — later guesses condition on earlier guesses, so errors compound. This is why full_acc falls faster than cond_acc.
num_speculative_tokens (K) is a tunable with a sweet spot: too few leaves speed unclaimed; too many wastes work on rejected guesses.

### 3. Quantization (FP8 dynamic)
FP8 = 8-bit weights (half of BF16) → half the memory traffic per token → ~2× headroom on the bottleneck. H100 has native FP8 cores, so no penalty.
"Dynamic" = activation scales computed at runtime → no calibration data needed (data-free, one-shot, ~15 min).
lm_head excluded because it decides next-token probabilities and is precision-sensitive.

### 4. How optimisations interact
The two wins are multiplicative (1802 > either alone), but the optimal K shifts when you quantise (K=2 BF16 → K=1 FP8): a cheaper verifier changes the cost/benefit of drafting. → Quantise first, then tune the draft head on the final verifier.

### 5. Training pipeline concepts
Offline vs online EAGLE-3 training: offline precomputes & stores all hidden states (1 GPU, lots of disk); online generates them live (needs ≥2 GPUs). You learned the trade-off and why offline fit your single H100.
Why hidden states are huge: thousands of floats × multiple layers per token vs a few integers for text.
Reading training metrics per draft position (val/loss_i, full_acc_i, cond_acc_i) instead of just total loss.

### 6. Benchmarking rigour (easy to underestimate)
Warm-up matters: the first run captures CUDA graphs / compiles — always discard it and measure the second. (Your cold runs: 1073/856/1704; warm: 1598/1287/1802.)
Fixed settings for comparability: same dataset, concurrency, seed, prefix-caching off — otherwise numbers aren't comparable to thresholds.
Tune with evidence: justify K using throughput + acceptance rate + acceptance length + TPOT together, not one number.

### 7. Practical MLOps skills
Isolated environments for conflicting toolchains (speculators / vLLM / llmcompressor) and debugging missing deps (vllm[bench], python3.12-dev for torch.compile).
GPU hygiene: killing leaked processes (pkill -f vllm) to avoid OOM-on-startup; reading nvidia-smi.
Cloud cost discipline: stop ≠ delete; disk + IP keep billing when stopped; capture results to disk at clean checkpoints so you can pause; clean up everything when done.
tmux for long jobs surviving SSH drops, and the serve-in-one-pane / bench-in-another pattern.


### The single biggest takeaway: on modern GPUs, LLM serving is memory-bandwidth bound, so the highest-leverage optimizations (quantization, speculative decoding) all reduce or amortize weight movement — and their ordering and tuning depend on how they affect that bottleneck.
