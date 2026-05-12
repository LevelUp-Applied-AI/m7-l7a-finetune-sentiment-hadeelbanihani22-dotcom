# Module 7 Week A — Lab Evaluation Report

## Dataset
1–2 sentences describing AARSynth app reviews Sentences-50Agree
(N examples, label distribution, split sizes).

## Model and hyperparameters
- Backbone: distilbert-base-uncased
- Number of labels: 3
- Learning rate, epochs, batch size, max_length, seed
- Training time (wall-clock) on your machine

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---|
| Accuracy | ... |
| Macro-F1 | ... |

Per class (read from `metrics.json`):

| Class | F1 | Precision | Recall |
|---|---|---|---|
| Positive | ... | ... | ... |
| Neutral  | ... | ... | ... |
| Negative | ... | ... | ... |

## Confusion matrix
[Insert a small markdown table or commit a screenshot. The starter
includes a helper that prints a sklearn confusion_matrix for you.]

## Three qualitative error examples (one per class)
For each example, include:
- The original sentence
- Gold label
- Predicted label
- Predicted probability for the gold label
- 1–2 sentences on why you think the model got this one wrong.

## Hugging Face Hub model URL
https://huggingface.co/HadeelBH/m7-app-review-sentiment