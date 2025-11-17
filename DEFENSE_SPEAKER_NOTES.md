DEFENSE SPEAKER NOTES — NutriSight (Memorization & Phrasing)

Use this file as speaker notes—short prompts with phrasing you'll use during the defense. Practice aloud 10x per item.

Format: [Question prompt] → Suggested 1–2 sentence phrasing → Memory cue

---

1. **Q:** What problem did you solve?

   **A:** "I built a unified classifier that recognizes 125 food classes and a non-food class for robust meal detection in a single pass, making deployment simpler and faster."

   _Cue:_ "Unified + single pass."

2. **Q:** Why a single model rather than 2-stage?

   **A:** "A single model reduces latency and complexity in production, removing the need for an extra binary filter and simplifying deployment and maintenance."

   _Cue:_ "Latency & simplicity."

3. **Q:** Why EfficientNet-B3?

   **A:** "It provides a high-accuracy architecture with a compact size (~12M params), making it practical for edge and server deployment with reasonable inference times."

   _Cue:_ "Accuracy + small size."

4. **Q:** Summarize training in one line

   **A:** "We use transfer learning with a short head warmup and then full fine-tuning, SGD with momentum, cosine LR, and gradient accumulation."

   _Cue:_ "Warmup → fine-tune, SGD."

5. **Q:** Why remove mixup/label smoothing/dropout?

   **A:** "On this dataset, heavy augmentation and a stable LR schedule worked better empirically; the regularizers reduced calibration and slightly hurt per-class performance."

   _Cue:_ "Augmentation > regularizers (here)."

6. **Q:** Test vs whole-dataset — how to explain to panel

   **A:** "The held-out test split (88.29%) is the key generalization measure; whole-dataset (96.21%) shows practical coverage, but it includes train/val so it's higher — use per-class metrics for a balanced view."

   _Cue:_ "Test=generalization; whole=coverage."

7. **Q:** Why Top-3 at 70% useful?

   **A:** "Top-3 at 70% captures situations where the correct answer is among the top predictions with sufficient confidence, which is practical for user confirmation workflows."

   _Cue:_ "Top-3 = user confirm."

8. **Q:** What are the weak points?

   **A:** "Five classes — mainly pork cuts and certain desserts — show lower F1 due to visual similarity; we can fix these with more data and targeted augmentations."

   _Cue:_ "Meat & dessert confusion."

9. **Q:** Non-food contamination explanation

   **A:** "Non-food detection is near-perfect, but raw produce on plain backgrounds sometimes triggers non-food scores, showing the model is sensitive to context."

   _Cue:_ "Raw fruit / context."

10. **Q:** How to improve low-performing classes, succinctly

    **A:** "Collect more varied examples for those classes, add context-aware augmentation, and consider class reweighting or ensemble models."

    _Cue:_ "More data, augmentation, reweighting."

11. **Q:** Deployment and performance

    **A:** "I export to ONNX; this allows fast inference in production. CPU runtimes ~50–150ms per image; quantization is the next step for mobile."

    _Cue:_ "ONNX, quantize for mobile."

12. **Q:** Reproducibility & evidence

    **A:** "All hyperparameters and results are logged; use `summary.json` and our notebooks for reproducible runs."

    _Cue:_ "summary.json + notebooks."

---

Practice tips (for you):
• Speak each phrasing out loud 10 times — memorize the cue and rehearse the 1–2 sentence phrasing.
• Keep answers short; follow up only if asked for more detail.
• When asked for numbers, cite quick stats: 88.29% Top-1 (test), 96.21% Top-1 (whole dataset), non-food ≈ 99.96%.

Optional: I can convert these notes into a printable speaker notes PDF or slides if you want. Add timings for each question (30–45s each) if you want a strict practice protocol.
