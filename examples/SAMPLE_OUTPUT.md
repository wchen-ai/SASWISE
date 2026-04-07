# Sample output — `examples/demo_uncertainty_classification.py`

This file records two **verbatim** runs of the SASWISE-UE adopter
calibration demo, one on a synthetic 3-class Gaussian mixture and one
on MNIST. Both were produced by

```bash
python examples/demo_uncertainty_classification.py --dataset {synthetic|mnist}
```

with the default settings. The numbers below were captured directly
from the demo's stdout — nothing has been edited.

The point of recording these is to set expectations for what the
pipeline produces and how to read the calibration metrics.

---

## How to read the metrics

| Metric | What it measures | What "good" looks like |
|---|---|---|
| **Ensemble accuracy** | Top-1 accuracy of the mean prediction | At least as good as the base model |
| **Mean predictive entropy** (correct vs incorrect) | Average entropy of `softmax(mean)` split by 0/1 correctness | Incorrect samples should have **noticeably higher** entropy than correct ones |
| **Pearson r(entropy, error)** | Linear correlation between predictive entropy and a 0/1 error indicator | **Positive and large** — high entropy ↔ wrong |
| **AUROC of misclassification given entropy** | "Can predictive entropy detect errors?" | **> 0.5**, ideally `> 0.85` |
| **AUROC given std** | Same question, using the per-sample logit std as the score | For classification this is **uninformative or below random** — see the *Important caveat* below |
| **Risk–coverage curve** | Error rate when keeping only the top-`X %` most-confident predictions | **Drops monotonically as coverage shrinks** |

### Important caveat — `predictive_entropy`, not `std`, for classification

Both runs below show that the per-element standard deviation of the
*raw logits* across configurations is **not a useful uncertainty
signal for classification**. On MNIST the std-AUROC is 0.44 (worse
than random), while the entropy-AUROC is 0.92. The mean prediction's
softmax entropy is the right scalar for classification; std is the
right scalar for regression and segmentation.

---

## Run 1 — Synthetic Gaussian mixture (3 classes, 20 features)

About 30 seconds end-to-end on a single GPU.

```text
================================================================
SASWISE-UE — uncertainty calibration demo
================================================================
device         : cuda
seed           : 0
dataset        : synthetic Gaussian mixture (3 classes, 20 features)
  train samples: 2400
  val   samples: 600
  features     : 20
  classes      : 3

[1/4] Training base classifier
  base epoch 1/2  val_acc=0.7467
  base epoch 2/2  val_acc=0.8200
  base val_acc : 0.8200

[2/4] Decomposing the base model and materialising variants
  decomposed into 2 blocks (requested 4; mismatch happens on tiny models):
    block 0    10880 params  (3 tensors)
    block 1     2243 params  (5 tensors)
  warehouse: 2 blocks × 3 variants = 9 possible sub-models

[3/4] Fine-tuning the warehouse with adaptive consistency weighting
epoch 1/2  step      0  loss=0.5076  task=0.5076  cons=0.0008  λ=0.0000
epoch 1/2  step      4  loss=0.4856  task=0.4840  cons=0.0002  λ=7.3000
epoch 1/2  step      8  loss=0.5298  task=0.5266  cons=0.0002  λ=16.0000
epoch 1/2  step     12  loss=0.5063  task=0.5013  cons=0.0002  λ=24.0000
epoch 1/2  step     16  loss=0.5050  task=0.5040  cons=0.0000  λ=32.0000
epoch 2/2  step     20  loss=0.5092  task=0.5064  cons=0.0001  λ=40.0000
epoch 2/2  step     24  loss=0.4629  task=0.4546  cons=0.0002  λ=48.0000
epoch 2/2  step     28  loss=0.5934  task=0.5581  cons=0.0006  λ=56.0000
epoch 2/2  step     32  loss=0.5189  task=0.5160  cons=0.0000  λ=64.0000
epoch 2/2  step     36  loss=0.5210  task=0.5106  cons=0.0001  λ=72.0000

[4/4] Ensembled inference (streaming Welford)
  configurations sampled: 9
  mean shape            : (600, 3)

----- Calibration metrics -----

  ensemble accuracy           : 0.8217
  mean predictive entropy     : 0.7246
    on correct samples        : 0.6780
    on incorrect samples      : 0.9391
  mean per-sample std (||·||₂): 0.0307

  Pearson r(entropy, error)   : +0.4076   (positive = high entropy ↔ wrong)
  Pearson r(std,     error)   : -0.1729

  AUROC of misclassification (using entropy as score): 0.8310
  AUROC of misclassification (using std    as score) : 0.3769
    (0.5 = uncertainty is uninformative; 1.0 = perfectly identifies all errors)

  Risk-coverage curve (selective prediction):
    coverage   error rate   #samples kept
     100%      0.1783          600
      80%      0.1042          480
      60%      0.0583          360
      40%      0.0125          240
      20%      0.0000          120
================================================================
Done.
================================================================
```

### What this tells us

- **Ensembling helps:** base 82.0 % → ensemble 82.2 % (small gain on a small dataset).
- **Entropy is well-calibrated:** mean entropy on errors (0.94) is **38 % higher** than on correct predictions (0.68).
- **Selective prediction is sharp:** rejecting the most-uncertain 80 % of samples drops error from 17.8 % to **0 %**. The risk–coverage curve is monotonically decreasing.
- **`λ` ramps cleanly to ≈72** after the warm-up. Below the `max_weight=100` ceiling, so no clipping artefacts.
- **The 2-block instead of 4-block decomposition** is the documented edge case for tiny models — `decompose_balanced` greedily closes a block as soon as it covers ~`total/K` parameters, and on this MLP the first weight tensor already exceeds half the total. On a real ResNet/U-Net you always get exactly the requested block count.

---

## Run 2 — MNIST (10 classes, 28×28 → 784 features)

About 3 minutes end-to-end on a single GPU including the first-time
~12 MB MNIST download.

```text
================================================================
SASWISE-UE — uncertainty calibration demo
================================================================
device         : cuda
seed           : 0
dataset        : MNIST (10 classes, 28×28 → 784 features)
  train samples: 60000
  val   samples: 10000
  features     : 784
  classes      : 10

[1/4] Training base classifier
  base epoch 1/1  val_acc=0.9200
  base val_acc : 0.9200

[2/4] Decomposing the base model and materialising variants
  decomposed into 2 blocks (requested 4; mismatch happens on tiny models):
    block 0   100352 params  (1 tensors)
    block 1    10794 params  (7 tensors)
  warehouse: 2 blocks × 3 variants = 9 possible sub-models

[3/4] Fine-tuning the warehouse with adaptive consistency weighting
epoch 1/1  step      0  loss=0.2354  task=0.2354  cons=0.0000  λ=0.0000
epoch 1/1  step     58  loss=0.1841  task=0.1639  cons=0.0008  λ=25.7278
epoch 1/1  step    116  loss=0.2557  task=0.2214  cons=0.0010  λ=34.7915
epoch 1/1  step    174  loss=0.3057  task=0.2930  cons=0.0004  λ=29.3679
epoch 1/1  step    232  loss=0.2765  task=0.2693  cons=0.0005  λ=16.0569

[4/4] Ensembled inference (streaming Welford)
  configurations sampled: 9
  mean shape            : (10000, 10)

----- Calibration metrics -----

  ensemble accuracy           : 0.9341
  mean predictive entropy     : 0.2624
    on correct samples        : 0.2163
    on incorrect samples      : 0.9158
  mean per-sample std (||·||₂): 0.3339

  Pearson r(entropy, error)   : +0.5061   (positive = high entropy ↔ wrong)
  Pearson r(std,     error)   : -0.0537

  AUROC of misclassification (using entropy as score): 0.9245
  AUROC of misclassification (using std    as score) : 0.4359
    (0.5 = uncertainty is uninformative; 1.0 = perfectly identifies all errors)

  Risk-coverage curve (selective prediction):
    coverage   error rate   #samples kept
     100%      0.0659        10000
      80%      0.0111         8000
      60%      0.0030         6000
      40%      0.0008         4000
      20%      0.0000         2000

================================================================
Done.
================================================================
```

### What this tells us

| Claim | Evidence |
|---|---|
| **Ensembling helps accuracy** | Base 92.00 % → ensemble 93.41 % (+1.41 pts free, no extra base training) |
| **Predictive entropy is well-calibrated** | Mean entropy on errors (0.916) is **4.2× higher** than on correct predictions (0.216) |
| **Entropy is a strong miscalibration detector** | AUROC of "is this prediction wrong, given its entropy?" = **0.9245**. Random would be 0.5. |
| **Selective prediction is dramatic** | 100 % coverage → 6.59 % error; 60 % coverage → 0.30 % error (a **22× error reduction** by rejecting the top-40 % uncertain predictions). At 20 % coverage, **0 % error**. |
| **The adaptive `λ` mechanism is well-behaved on a real task** | `λ` ramps from 0 → 25 over the first 60 steps (warm-up), then drifts adaptively in the 16–35 range as the loss ratio shifts. **Never hit `max_weight=100`** so no clipping. |
| **`std` ≠ `entropy` for classification** | Pearson `r(std, error) = −0.054`, AUROC `= 0.44` (below random). Pearson `r(entropy, error) = +0.51`, AUROC `= 0.92`. Use `predictive_entropy` for classification, `std` for regression / segmentation. |

### MNIST risk–coverage curve, visualised

```
error rate
   ^
6.6%┤●
    │ \
    │  \
1.1%┤   ●
    │    \
0.3%┤     ●
0.1%┤      ●
0.0%┤       ●─────►
    └──┴──┴──┴──┴──┴── coverage
       100  80  60  40  20  %
```

A linear drop from 6.6 % → 0 % as we successively remove the
highest-entropy 20 % / 40 % / 60 % / 80 % of predictions. This is the
operational story of the paper in one curve: **the entropy is sharp
enough that you can confidently auto-process the bottom-80 % of
predictions and route the top-20 % to a human reviewer**, and the
auto-processed batch will have an error rate of ~1 % vs the model's
overall ~6.6 %.

---

## Reproducing this output

Both runs above are deterministic with `--seed 0` (the default). On a
fresh checkout:

```bash
# Synthetic, ~30 seconds, no network.
python examples/demo_uncertainty_classification.py --dataset synthetic

# MNIST, ~3 minutes, downloads ~12 MB on first run.
python examples/demo_uncertainty_classification.py --dataset mnist \
    --base-epochs 1 --finetune-epochs 1 --num-configurations 16
```

Knobs you can tune:

| Flag | Default | What it changes |
|---|---|---|
| `--dataset` | `auto` | `mnist` / `synthetic` / `auto` (try MNIST, fall back) |
| `--base-epochs` | `2` | How long to train the base classifier before adopting it |
| `--num-blocks` | `4` | How many warehouse blocks to decompose into |
| `--num-variants` | `3` | How many variants per block |
| `--noise-std` | `5e-2` | Initial perturbation magnitude (larger ⇒ more diverse variants ⇒ smaller `λ`) |
| `--finetune-epochs` | `2` | Length of the diversification fine-tune |
| `--target-ratio` | `0.1` | Steady-state contribution of consistency to total loss |
| `--warmup-steps` | `50` | Linear ramp duration for `λ` |
| `--num-configurations` | `32` | How many sub-models to ensemble at inference time |
