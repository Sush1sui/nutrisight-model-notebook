DEFENSE SPEAKER NOTES — NutriSight (Memorization & Phrasing)

Use this file as speaker notes—short prompts with phrasing you'll use during the defense. Practice aloud 10x per item.

Format: [Question prompt] → Suggested 1–2 sentence phrasing → Memory cue

---

1. What problem did you solve?
   Phrasing: "I built a unified classifier that recognizes 125 food classes and a non-food class for robust meal detection in a single pass, making deployment simpler and faster."
   Cue: "Unified + single pass."

2. Why a single model rather than 2-stage?
   Phrasing: "A single model reduces latency and complexity in production, removing the need for an extra binary filter and simplifying deployment and maintenance."
   Cue: "Latency & simplicity."

3. Why EfficientNet-B3?
   Phrasing: "It provides a high-accuracy architecture with a compact size (~12M params), making it practical for edge and server deployment with reasonable inference times."
   Cue: "Accuracy + small size."

4. Summarize training in one line
   Phrasing: "We use transfer learning with a short head warmup and then full fine-tuning, SGD with momentum, cosine LR, and gradient accumulation."
   Cue: "Warmup → fine-tune, SGD."

5. Why remove mixup/label smoothing/dropout?
   Phrasing: "On this dataset, heavy augmentation and a stable LR schedule worked better empirically; the regularizers reduced calibration and slightly hurt per-class performance."
   Cue: "Augmentation > regularizers (here)."

6. Test vs whole-dataset — how to explain to panel
   Phrasing: "The held-out test split (88.29%) is the key generalization measure; whole-dataset (96.21%) shows practical coverage, but it includes train/val so it's higher — use per-class metrics for a balanced view."
   Cue: "Test=generalization; whole=coverage."

7. Why Top-3 at 70% useful?
   Phrasing: "Top-3 at 70% captures situations where the correct answer is among the top predictions with sufficient confidence, which is practical for user confirmation workflows."
   Cue: "Top-3 = user confirm."

8. What are the weak points?
   Phrasing: "Five classes — mainly pork cuts and certain desserts — show lower F1 due to visual similarity; we can fix these with more data and targeted augmentations."
   Cue: "Meat & dessert confusion."

9. Non-food contamination explanation
   Phrasing: "Non-food detection is near-perfect, but raw produce on plain backgrounds sometimes triggers non-food scores, showing the model is sensitive to context."
   Cue: "Raw fruit / context."

10. How to improve low-performing classes, succinctly
    Phrasing: "Collect more varied examples for those classes, add context-aware augmentation, and consider class reweighting or ensemble models."
    Cue: "More data, augmentation, reweighting."

11. Deployment and performance
    Phrasing: "I export to ONNX; this allows fast inference in production. CPU runtimes ~50–150ms per image; quantization is the next step for mobile."
    Cue: "ONNX, quantize for mobile."

12. Reproducibility & evidence
    Phrasing: "All hyperparameters and results are logged; use `summary.json` and our notebooks for reproducible runs."
    Cue: "summary.json + notebooks."

---

Practice tips (for you):
• Speak each phrasing out loud 10 times — memorize the cue and rehearse the 1–2 sentence phrasing.
• Keep answers short; follow up only if asked for more detail.
• When asked for numbers, cite quick stats: 88.29% Top-1 (test), 96.21% Top-1 (whole dataset), non-food ≈ 99.96%.

Optional: I can convert these notes into a printable speaker notes PDF or slides if you want. Add timings for each question (30–45s each) if you want a strict practice protocol.
