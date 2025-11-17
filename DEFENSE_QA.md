# Thesis Defense Q&A — NutriSight (Panel-Friendly Short Answers)

Quick, crisp answers to practice for your thesis final defense. Each answer is short and tailored for quick recall.

---

## Quick Facts

- Model: EfficientNet-B3 (12M params).
- Classes: 126 (125 food + 1 non-food).
- Key artifacts: `summary.json`, `classification_report.txt`, `model.onnx`, evaluation notebooks.

---

## High-Level

Q: What problem did you solve?
A: I built a single model that recognizes many food types and rejects non-food, to automate meal recognition.

Q: Why a single model not two-stage?
A: Simpler and faster at runtime — one pass gives both food classification and non-food rejection.

Q: What's new in this work?
A: A 126-class dataset; a practical transfer-learning pipeline; production-ready ONNX model and full-dataset validation tools.

---

## Data

Q: Dataset size and split?
A: 53,918 images across 126 classes; ~350 images per food class and 10,168 non-food images. Stratified train/val/test splits.

Q: Why a large non-food set?
A: To cover diverse negatives and reduce false positives in real-world deployment.

---

## Model & Training

Q: Why EfficientNet-B3?
A: Balanced accuracy and size — good for deployment on standard hardware.

Q: How was it trained?
A: Transfer learning: quick head warmup, then full-model fine-tuning; SGD with momentum and Cosine LR; batch size 16 with gradient accumulation.

Q: Any special regularization choices?
A: We used strong augmentation and removed mixup/label-smoothing/dropout after empirical testing; this gave better per-class performance.

---

## Metrics & Evaluation

Q: Where are the metrics?
A: See `summary.json` for run metrics and `per_class_metrics.json` / `classification_report.txt` for per-class scores.

Q: Test vs whole-dataset results — what's the difference?
A: Test is held-out: Top-1 88.29%. Whole-dataset includes train/val so it's higher: Top-1 96.21%. Use macro F1 for a balanced view.

Q: What is Top-3 @70%?
A: If the correct label appears in top-3 with ≥70% confidence, we count it as a useful result for user confirmation.

---

## Results & Findings

Q: Headline performance?
A: Test Top-1 88.29%, Top-5 97.15%, Macro F1 87.71%.

Q: Which classes need work?
A: `pork_bistek`, `pork_chop`, `steak`, `tiramisu`, `chocolate_mousse` — often confused with similar items.

---

## Confusion Matrix (Plain English)

Q: How are confusion matrices built and top confusions found?
A: We compare each predicted label with the true label and count occurrences. Normalizing per true class shows percentages. We sort the off-diagonal counts/percentages and list the largest to find frequent misclassifications.

---

## Deployment & Reproducibility

Q: Export and inference speed?
A: ONNX export (`model.onnx`). On CPU ~50–150ms/image; GPU faster. Model is ~45MB.

Q: How to reproduce?
A: Run `train_efficientnet_b3_optimized.ipynb` and `test_126class_model_on_all_datasets.ipynb`; update file paths (ckpt and dataset) and run cells. `summary.json` records config.

---

## Defense-Focused Short Answers

Q: Does the model generalize?
A: Yes — 88.29% on the held-out test split demonstrates generalization; whole-dataset metrics confirm practical coverage.

Q: Should the non-food class be balanced?
A: It’s intentionally large for negative coverage, but balancing or weighting can be tested if aggregation bias becomes an issue.

Q: Why not do detection for multi-food images?
A: Detection/segmentation is the next step — this thesis focuses on single-label classification to keep scope manageable.

Q: What are your next steps if you had more compute?
A: Try larger models, ensemble methods, add detection for multi-food, and collect more examples for challenging classes.

---

If you want, I can convert this to speaker notes, a one-page cheat sheet, or flash-card prompts for fast practice.

# Thesis Defense Q&A — NutriSight (Short Answers)

This file contains short, panel-friendly answers for expected questions about the 126-class unified model and evaluation.

---

## Quick Reference

- Model: EfficientNet-B3 (12M parameters), exported as ONNX (`model.onnx`).
- Run folder: `efficientnet_b3_baseline-20251116-101749/` (includes `summary.json`, `classification_report.txt`).
- Full validation: `inference_test_on_whole_dataset/` (`inference_report.txt`, `summary.txt`).
- Key notebooks: `train_efficientnet_b3_optimized.ipynb`, `test_126class_model_on_all_datasets.ipynb`, `confusion_matrix_all_classes.ipynb`.

---

## High-Level Questions

Q: What problem does your thesis address?
A: Build a single model that recognizes many food types and can reject non-food images — useful for automated meal detection.

Q: Why a single model instead of two steps?
A: Simpler, faster, easier to deploy — fewer steps means lower latency and easier maintenance.

Q: What are the main contributions?
A: The 126-class dataset, an optimized transfer-learning training pipeline, a production-ready ONNX model, and full-dataset evaluation tools.

---

## Dataset & Splits

Q: How large is the dataset and what is the class balance?
A: 53,918 images in 126 classes. Food classes have ~350 images each; non-food has 10,168 images by design to cover many negative examples.

Q: Why so many non-food images?
A: To improve robustness of non-food rejection, i.e., minimizing false positives in real-world images.

---

## Model & Training

Q: Why EfficientNet-B3?
A: Good accuracy in a compact model — practical for edge and mobile deployment.

Q: How was the model trained?
A: Transfer learning — short head warmup then full fine-tuning. SGD with momentum and Cosine annealing, batch size 16 with gradient accumulation for a larger effective batch size.

Q: Why remove advanced regularizations (mixup, label smoothing, dropout)?
A: Experimentally, strong augmentation + stable LR schedule gave better per-class performance; the regularizers reduced calibration for some classes in this dataset.

---

## Metrics & Evaluation

Q: Where are the metrics stored?
A: `efficientnet_b3_baseline-20251116-101749/summary.json` and per-class files like `per_class_metrics.json` and `classification_report.txt`.

Q: What's the difference between test split and whole-dataset metrics?
A: The test split is held-out and shows generalization (Top-1 88.29%). Whole-dataset includes train and val, so it can be higher because it has more similar images (Top-1 96.21%). Use macro F1 for a fair per-class view.

Q: What is Top-3 at 70%?
A: If one of the top-3 predictions has >=70% confidence, we count it as a strong multi-hypothesis success — helpful for user confirmation flows.

---

## Results & Findings

Q: What are the headline results (test split)?
A: Top-1: 88.29%; Top-5: 97.15%; Macro F1: 87.71%; Weighted F1: 88.16%.

Q: What does whole-dataset testing show?
A: 96.21% Top-1 across 53,918 images, largely due to many non-food images which the model identifies accurately.

Q: Which classes remain challenging?
A: `pork_bistek`, `pork_chop`, `steak`, `tiramisu`, `chocolate_mousse` — mostly visually similar or presentation-variable.

---

## Confusion Matrix & Error Analysis

Q: How are confusion matrices computed? (Plain language)
A: For each image we check the model's predicted label and compare it with the true label. We count how often each true label was predicted as each possible label — that table of counts is the confusion matrix. We convert to percentages (per true class) to understand error rates.

Q: How are the worst confusions found?
A: We ignore correct predictions and sort the remaining pairs by how often they occur (either by counts or percentage). The top pairs are the most frequent mistakes and show where to add data or improve the model.

Q: Why show a "Top-20" confusion matrix in the paper?
A: The full 126×126 matrix is too dense to read in a publication, so we include a Top‑20 subset that highlights the most meaningful confusions — the complete matrix is available in the repo for deeper analysis.

---

## Deployment & Reproducibility

Q: How is the model exported and what is expected runtime?
A: Exported as ONNX (`model.onnx`). On a typical CPU it can do 50–150ms per image; GPU will be faster. Model is ~45MB.

Q: How can someone reproduce your runs?
A: Use `train_efficientnet_b3_optimized.ipynb` to train and `test_126class_model_on_all_datasets.ipynb` to evaluate. Update `ckpt_path` and `root_dir`, then run cells. `summary.json` lists hyperparameters used.

---

## Tough Defense Questions (Short answers)

Q: "Whole-dataset accuracy is high — does it mean your model generalizes?"
A: The held-out test split (88.29%) is the formal generalization metric. Whole-dataset confirms practical coverage but is not a replacement for held-out evaluation.

Q: "Should you rebalance the non-food class?"
A: Possible — the large non-food set improves negative coverage but could bias aggregate metrics. We use macro metrics and per-class reports to avoid misleading conclusions.

Q: "Why not use object detection for multiple foods per image?"
A: Multi-food images require detection or segmentation; that’s a future extension. The current scope is single-label classification per image for simplicity and clarity.

Q: "If you had more time or compute, what changes would you make?"
A: Add larger models, test ensembles, expand low-performing classes with more data, and explore detection/multi-label workflows.

---

## Quick file references

- `efficientnet_b3_baseline-20251116-101749/summary.json`
- `classification_report.txt` and `per_class_metrics.json`
- `inference_test_on_whole_dataset/inference_report.txt`
- `test_126class_model_on_all_datasets.ipynb` and `confusion_matrix_all_classes.ipynb`
