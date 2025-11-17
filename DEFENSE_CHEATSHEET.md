# Defense Cheatsheet — Top 12 Quick Answers

1. **Q:** What is the main innovation?

   **A:** One model recognizes 125 foods and also correctly rejects non-food — so we don't need a separate filter.

2. **Q:** Key metrics (test split)?

   **A:** Top-1: 88.29% | Top-5: 97.15% | Macro F1: 87.71%.

3. **Q:** Whole dataset metric (all splits)?

   **A:** Top-1: 96.21% (53,918 images) — includes many non-food examples; use macro metrics to avoid bias.

4. **Q:** What architecture and tools?

   **A:** EfficientNet-B3, PyTorch + DirectML; exported to ONNX for deployment.

5. **Q:** Why EfficientNet-B3?

   **A:** Best compromise: high accuracy, moderate size (~12M params), and faster inference suitable for edge deployment.

6. **Q:** Training strategy?

   **A:** Two-phase transfer learning: quick head warmup, then fine-tune whole network. We used SGD with momentum and a cosine LR schedule; gradient accumulation to simulate larger batches.

7. **Q:** Why no mixup/label smoothing/dropout?

   **A:** They didn’t help for this dataset; strong augmentations plus the scheduler gave better per-class results.

8. **Q:** Biggest weaknesses?

   **A:** 5 classes with F1 < 70%: `pork_bistek`, `pork_chop`, `steak`, `tiramisu`, `chocolate_mousse` — mostly visual similarity and presentation variations.

9. **Q:** Non-food performance & contamination?

   **A:** Non-food accuracy: ~100% on the full set. The main problem is that raw fruits can sometimes be misclassified due to contextual cues (e.g., on white backgrounds they look like non-food).

10. **Q:** Confusion matrix (short note)

    **A:** We show a condensed 'Top‑20' confusion matrix in the paper to highlight the key errors; the full 126×126 matrix is available in the repo for complete analysis.

11. **Q:** How to improve poor classes?

    **A:** More targeted data collection (diverse examples), class-weighting or focal loss, context-aware augmentation, or ensemble/architectural upgrades.

12. **Q:** How reproducible is the work?

    **A:** All hyperparameters and artifact paths are in `summary.json`. Repro via `train_efficientnet_b3_optimized.ipynb` and `test_126class_model_on_all_datasets.ipynb`.

13. **Q:** How to deploy?

    **A:** Exported ONNX (`model.onnx`). Use ONNX runtime; quantize if needed for mobile. CPU inference ~50-150ms; GPU runs faster.

**Files to reference in defense**: `README.md`, `summary.json`, `classification_report.txt`, `inference_test_on_whole_dataset/inference_report.txt`, `test_126class_model_on_all_datasets.ipynb`.

---

Use this as your quick recall cheat sheet right before questions. For deeper followups refer to `DEFENSE_QA.md` and the notebooks.
