# Deep Learning & Big Data — Project Documentation

## Pet Breed & Species Recognition on the Oxford-IIIT Pet Dataset

*Course: AIN · Two Models, Two Tasks, One Dataset*

---

## 1. Problem Statement

The goal of this project is to understand, implement, and critically compare deep learning models on a single image dataset. We use the Oxford-IIIT Pet dataset and solve two distinct supervised tasks on the same images, with two fundamentally different modelling approaches, then compare them on identical metrics.

**Two tasks on the same dataset.** We address (A) multi-class classification of the pet breed (37 classes) and (B) binary classification of the species (cat vs. dog). Both tasks read exactly the same images; only the target label differs. This satisfies the assignment's "two tasks on the same dataset" option (multi-class + binary classification).

**Two models.** Model 1 is a convolutional neural network implemented and trained entirely from scratch. Model 2 uses transfer learning: a ResNet-18 pretrained on ImageNet, fine-tuned on our data. Comparing a from-scratch network with a pretrained backbone on the same tasks is the central experiment.

---

## 2. Dataset Description

The Oxford-IIIT Pet dataset contains roughly 7,349 images spanning 37 pet breeds — 12 cat breeds and 25 dog breeds — with about 200 images per breed. Images vary widely in scale, pose, and lighting, which makes the breed task genuinely challenging.

We access the dataset through torchvision's built-in `OxfordIIITPet` class, which can return two labels simultaneously for every image: a breed label (`category`, an integer 0–36) and a species label (`binary-category`, 0 for cat and 1 for dog). This dual-label capability is precisely why the dataset is a natural fit for the two-tasks requirement — a single sample carries both targets, so both tasks operate on identical inputs.

**Splits.** We use the official `trainval` split (3,680 images) for training and validation, and the official `test` split (3,669 images) as a held-out test set used only for final evaluation. From `trainval` we carve out 15% as a validation set using a fixed, seeded permutation of indices, so the split is identical across every experiment.

**Class balance.** Breeds are roughly balanced (~150 train images each). The species task, however, is imbalanced at approximately 2:1 dogs to cats, because there are more dog breeds. We therefore report macro-F1 alongside accuracy so that the minority class is not masked by the majority.

**Preprocessing & augmentation.** Training images receive light augmentation — random horizontal flip, small rotation, and (for the scratch CNN) colour jitter — to enlarge effective data variety and reduce overfitting on the small per-class counts. Validation and test images use deterministic resize + normalise only. The from-scratch CNN uses 128×128 inputs with generic 0.5/0.5 normalisation; the ResNet uses 224×224 inputs normalised with ImageNet statistics, matching what the pretrained weights expect.

---

## 3. Model 1 — CNN From Scratch

Model 1 is a compact VGG-style convolutional network built and trained from zero. The same backbone serves both tasks; only the final layer's width changes (37 outputs for breed, 2 for species), which isolates the task as the single variable.

### 3.1 Architecture and design justification

- **Four convolutional blocks** with widths 32 → 64 → 128 → 256. Each block is Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → MaxPool. Stacking small 3×3 convolutions is parameter-efficient and grows the receptive field gradually; doubling channels as spatial resolution halves is the standard VGG pattern that keeps computation balanced across depth.
- **Batch Normalization** after every convolution stabilises and accelerates training and provides mild regularisation — valuable when training from scratch on limited data.
- **ReLU activations** because they are cheap, non-saturating, and the well-established default for CNNs.
- **Max pooling** halves spatial size per block, reducing compute while adding a small amount of translation invariance.
- **Global Average Pooling** replaces a large flatten-plus-dense layer. This drastically reduces parameters (limiting overfitting) and makes the head robust to input size.
- **Dropout (p = 0.4)** before the final linear layer adds further regularisation.
- **Linear classification head** whose output dimension is set per task. The network has roughly 1.18 million parameters — small enough to train from scratch quickly.

### 3.2 Training configuration and why

- **Loss — Cross-Entropy.** Standard for single-label classification. For the binary species task we use a 2-logit softmax head with cross-entropy, which is equivalent to sigmoid + binary cross-entropy but keeps the code identical across tasks.
- **Optimizer — Adam.** An adaptive optimizer that converges reliably with little tuning, a sensible default for a from-scratch model.
- **Weight decay (1e-4).** Light L2 regularisation to discourage overly large weights.
- **LR schedule — StepLR.** The learning rate decays in steps so that later epochs fine-tune rather than oscillate around the optimum.
- **Model selection.** We keep the weights from the epoch with the best validation accuracy, an early-stopping-style safeguard against late-epoch overfitting.

---

## 4. Model 2 — Transfer Learning (ResNet-18)

Model 2 reuses a ResNet-18 whose convolutional layers were pretrained on ImageNet's 1.2 million images. Because pets are well represented in ImageNet, the low- and mid-level features (edges, textures, fur, ears) transfer directly to our task.

### 4.1 Approach and justification

- **Pretrained backbone.** We load ResNet-18 with its ImageNet weights instead of random initialisation, so the model starts from useful visual features rather than noise.
- **New classification head.** We replace the final fully-connected layer with a fresh linear layer sized to the task (37 or 2). Only this head is new; the rest inherits pretrained weights.
- **Full fine-tuning.** We train the entire network (not just the head) at a modest learning rate. This is affordable because ResNet-18 is small, and it pays off because the dataset is similar enough to ImageNet to benefit from adapting all layers.
- **Same training engine.** We use the identical loss, optimizer, scheduler, and model-selection logic as Model 1, so any performance difference reflects the modelling approach rather than the training procedure. Transfer learning needs far fewer epochs to converge.

---

## 5. Results

All four trained models (two models × two tasks) are evaluated on the held-out official test split, which is never used for training or model selection. We report accuracy and macro-F1. The table below shows the expected pattern; exact numbers depend on the training run, but the relative ordering is robust.

| Model            | Task              | Accuracy (≈) | Macro-F1 (≈) |
| ---------------- | ----------------- | ------------ | ------------ |
| Scratch CNN      | Breed (37-class)  | 0.40 – 0.55  | 0.38 – 0.53  |
| Scratch CNN      | Species (binary)  | 0.90 – 0.95  | 0.89 – 0.95  |
| ResNet-18 (TL)   | Breed (37-class)  | 0.88 – 0.93  | 0.88 – 0.93  |
| ResNet-18 (TL)   | Species (binary)  | 0.98 – 0.99  | 0.98 – 0.99  |

*Ranges above are typical published/observed results for this dataset and architecture class; record your own run's exact figures from the notebook's results table.*

**Visual outputs in the notebook.** The notebook also produces training/validation curves for every model, a side-by-side accuracy bar chart, 2×2 confusion matrices for the species task, a 37×37 confusion heatmap for the breed task, and a per-breed precision/recall/F1 report for the strongest model.

---

## 6. Comparison

- **Transfer learning wins decisively on the breed task.** The from-scratch CNN must learn all visual features from only ~3,000 training images, whereas ResNet-18 reuses ImageNet features and merely adapts. This gap is the headline finding.
- **Both models handle the species task well.** Cat-vs-dog is far easier than fine-grained breed recognition, so even the small from-scratch CNN scores highly; the transfer model still edges ahead and converges in only a few epochs.
- **Efficiency favours transfer learning.** ResNet-18 reaches its best accuracy in a handful of epochs, while the scratch CNN needs many more epochs and still plateaus lower — transfer learning is both more accurate and more compute-efficient here.

---

## 7. Discussion and Limitations

**Why the breed task is hard.** It is a fine-grained problem: many breeds differ only in subtle fur patterns or face shapes, and there are only ~150 training images per class. The breed confusion matrix typically shows errors clustered among visually similar breeds.

- **Small dataset.** ~150 images per breed limit how well a from-scratch model can generalise; more data or stronger augmentation (MixUp, RandAugment) would help.
- **No hyperparameter search.** We used sensible defaults; a sweep over learning rate, image size, and augmentation strength would likely raise both models.
- **Single split and seed for reported numbers.** A publication-grade result would use k-fold cross-validation or multiple seeds and report mean ± standard deviation.
- **Species class imbalance (~2:1).** Accuracy alone can mislead on the binary task, which is why we also report macro-F1.
- **Compute scope.** Larger backbones (ResNet-50, ViT) would probably push breed accuracy higher but were out of scope.

**Reproducibility.** All random seeds are fixed and the train/validation split is deterministic, so re-running the notebook with the same configuration reproduces the reported numbers (up to minor GPU cuDNN nondeterminism).

---

## 8. How to Run the Notebook

- Open `Oxford_Pets_DeepLearning_Project.ipynb` in Jupyter or Google Colab (a GPU runtime is recommended but not required).
- Run cells top to bottom. The first run downloads the dataset (~800 MB) automatically via torchvision.
- For a quick end-to-end smoke test, set `CONFIG["fast_dev"] = True` (tiny subset, few epochs). For full results, leave it `False`.
- The final cell saves the best model weights and a `results.json` into a `checkpoints/` folder for reuse in the presentation.

---

*Mapping to the deliverables: Section 1 = problem statement; Section 2 = dataset description; Section 3 = Model 1; Section 4 = Model 2; Section 5 = results; Section 6 = comparison; Section 7 = discussion and limitations.*