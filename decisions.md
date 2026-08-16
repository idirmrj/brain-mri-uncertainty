## Why this project

I wanted a project outside of LLMs. My internships and academic work were mostly
NLP / RAG, so my profile looked one-sided. This project shows I can do computer
vision and deep learning end to end.

I deliberately took an **academic project I had already done** "brain tumor
detection" and rebuilt it properly. Before, it was a plain classifier: image in,
label out, "94% accuracy", and that was it. Here I turned it into something that
behaves more like a real medical tool: it doesn't just predict, it says **how sure
it is**, and it flags the cases it isn't sure about. That upgrade is the whole
point.

## The problem

A model that just says "tumor, 95%" isn't usable on its own in medicine. A doctor
needs to know **when the model is unsure**, so the doubtful cases get a second look
instead of being trusted blindly. A confident-but-wrong model is dangerous.

## The solution

A classifier that, on top of the prediction, gives an **uncertainty score**, and
sends the most uncertain cases to a human. Framed as a **triage aid, not a
diagnosis tool**.

## Dataset choice

- Used the **Brain Tumor MRI dataset (Kaggle, 7,023 images, 4 classes)**, the
  most common one in recent papers, so my results are comparable, and it's clean
  and ready to use.
- It's **balanced** (1,400 images per class), which I checked in exploration. So I
  used plain cross-entropy, no class weighting needed.
- **Limits I'm aware of:** no acquisition metadata (scanner, sequences), so I can't
  test how the model holds up on images from a different hospital (domain shift).
  The images are curated and clean; real clinical data is noisier. This is exactly
  why I built the uncertainty part on an unusual image, the model should say
  "I'm not sure" instead of guessing confidently.

## Preprocessing

- Resize to **224x224** (standard for an ImageNet-pretrained encoder).
- **Convert everything to RGB.** Exploration showed the dataset mixes grayscale and
  RGB images; converting to RGB makes them consistent and matches the pretrained
  encoder (which expects 3 channels).
- **ImageNet normalization**, to match the pretrained encoder.
- **Gentle augmentation**: horizontal flip, small rotations/shifts. No vertical
  flips or big rotations they'd create anatomically impossible brains.

## Model architecture

- **ResNet (pretrained on ImageNet) as the encoder + a small head with dropout.**
  Transfer learning: the encoder already knows edges and textures, I don't relearn
  that from scratch on a small dataset.
- The **dropout in the head is the key piece**: I keep it active at inference to
  measure uncertainty (see below).

## How I measure uncertainty (MC Dropout)

- I run the same image through the model **30 times with dropout on**. Each pass
  drops different neurons, so I get 30 slightly different answers.
- **Average of the 30** = the prediction. **Spread of the 30** = the uncertainty.
- Intuition: on an easy case the model agrees with itself every time (low spread);
  on a hard case it wavers (high spread).
- **Why this method:** it's the cheapest option; no extra model, I reuse the
  dropout that's already there, one training run. Alternatives: deep ensembles
  (train several models, usually a bit better but several times the cost) and
  plain softmax (but that's overconfident and unreliable).
- **I checked it actually works:** errors are ~11x more uncertain than correct
  predictions. So the uncertainty is a real, usable signal.

## Rejection (triage)

- I sort predictions by uncertainty and **hand the most uncertain ones to a human**.
- Instead of picking one arbitrary threshold, I plot the whole **risk-coverage
  curve** so the operating point can be chosen for the clinical use.
- Result: rejecting the ~20% most uncertain cases drops the error from **4.9% to
  under 0.4%**.

## Calibration

- I checked whether the confidence is trustworthy (does "90% sure" mean right 90%
  of the time?) using **ECE** and a reliability diagram, and corrected it with
  **temperature scaling**.
- The model was **already well calibrated** (ECE 0.038), so the fix only helped
  slightly. That's a fine result the point is that I can measure and correct
  calibration.

## Grad-CAM

- Shows **where the model looked** to decide. Used both as a visual and as a
  sanity check.
- Honest read: it localizes focal tumors (like meningiomas) well; it's more
  diffuse on infiltrating gliomas and on no-tumor cases which makes sense given
  those classes.

## What I compared (in MLflow)

- ResNet18 vs ResNet34
- Dropout 0.3 / 0.5 / 0.7 (directly affects the uncertainty)
- Frozen vs fine-tuned encoder
- MC Dropout vs a deep ensemble

## Limitations (assumed, not hidden)

- Single dataset, curated and clean: no domain-shift testing.
- Triage aid, **not** a diagnostic tool; would need clinical validation.
- MC Dropout can underestimate uncertainty compared to a full ensemble.
