# Module 7 Week A — Lab Evaluation Report

## Dataset

The dataset used for this lab is the AARSynth app reviews dataset, which contains app-review sentences labeled into three sentiment classes: negative, neutral, and positive. The full dataset contains 7,472 examples and was split into an internal train/test split using an 80/20 split.

The test split contains 1,495 examples:
- Negative: 499
- Neutral: 463
- Positive: 533

## Model and hyperparameters

- Backbone: `distilbert-base-uncased`
- Number of labels: 3
- Labels: negative, neutral, positive
- Learning rate: 5e-5
- Epochs: 2
- Batch size: 8
- Max sequence length: 128
- Seed: 42
- Training time: Not recorded

## Metrics on the test split

Aggregate:

| Metric | Value |
|---|---:|
| Accuracy | 0.6294 |
| Macro-F1 | 0.6276 |

Per class:

| Class | F1 | Precision | Recall |
|---|---:|---:|---:|
| Negative | 0.7159 | 0.7224 | 0.7094 |
| Neutral | 0.4788 | 0.4585 | 0.5011 |
| Positive | 0.6880 | 0.7114 | 0.6660 |

## Confusion matrix

Rows represent the true label, and columns represent the predicted label.

| True \ Predicted | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Negative | 354 | 127 | 18 |
| Neutral | 105 | 232 | 126 |
| Positive | 31 | 147 | 355 |

## Three qualitative error examples

### Error example 1 — Gold label: Negative

- Original sentence: `plz update it it stick in the loading screen forever`
- Gold label: negative
- Predicted label: neutral
- Predicted probability for the gold label: 0.3379

The model likely missed the negative sentiment because the sentence is written as a request to update or fix the app, rather than using strong negative words. Although “stick in the loading screen forever” indicates a serious issue, the model treated it as a neutral technical request.

### Error example 2 — Gold label: Neutral

- Original sentence: `nice app to use with friends`
- Gold label: neutral
- Predicted label: positive
- Predicted probability for the gold label: 0.0917

The model likely predicted positive because the phrase “nice app” strongly signals positive sentiment. However, the gold label is neutral, possibly because the review is short and does not express strong enthusiasm or detailed positive evaluation.

### Error example 3 — Gold label: Positive

- Original sentence: `good, but slow workflow.`
- Gold label: positive
- Predicted label: neutral
- Predicted probability for the gold label: 0.3745

The model likely got this wrong because the sentence contains mixed sentiment. The word “good” is positive, but “slow workflow” introduces a limitation, so the model classified it as neutral rather than positive.

## Hugging Face Hub model URL

https://huggingface.co/HadeelBH/m7-app-review-sentiment