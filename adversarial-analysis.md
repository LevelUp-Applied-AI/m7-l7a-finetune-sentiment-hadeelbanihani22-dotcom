# Adversarial Evaluation Analysis

## Per-hypothesis accuracy

I ran the fine-tuned Lab 7A app-review sentiment classifier against a hand-crafted adversarial set of 30 examples. The overall accuracy was **16 correct predictions out of 30 examples**, or **53.3%**.

| Hypothesis category | Correct | Total | Accuracy |
|---|---:|---:|---:|
| negation | 2 | 5 | 40.0% |
| lexical_trigger | 2 | 5 | 40.0% |
| domain_shift | 4 | 5 | 80.0% |
| length_extreme | 4 | 5 | 80.0% |
| sarcasm | 0 | 5 | 0.0% |
| other | 4 | 5 | 80.0% |
| overall | 16 | 30 | 53.3% |

The model performed best on `domain_shift`, `length_extreme`, and `other`, with 80.0% accuracy in each category. It struggled with `negation` and `lexical_trigger`, with 40.0% accuracy in each category. The clearest failure mode was `sarcasm`, where the model achieved 0.0% accuracy.

## Confirmed hypotheses

The strongest confirmed hypothesis is that the model struggles with sarcasm. It failed all five sarcasm examples. This suggests that the model often follows the surface polarity of cue words such as "Great", "Amazing", "Perfect", "Love", and "Fantastic" rather than inferring the intended negative meaning from the full sentence.

For example, row 21 says, "Great, another update that broke everything." A human would label this as negative because the word "Great" is sarcastic, but the actual complaint is that the update broke everything. This kind of example tests whether the classifier understands pragmatic meaning rather than just individual positive words.

The negation hypothesis was also partially confirmed. The model only achieved 40.0% accuracy on negation examples. This means it sometimes missed the effect of words such as "not", "does not", or "cannot" when those words changed the sentiment of a strong cue word. For example, row 2 says, "This app is not bad after the update." The expected label is positive, but a model that overweights the word "bad" may classify it as negative.

The lexical_trigger hypothesis was also confirmed. The model achieved only 40.0% accuracy on examples that contain strong sentiment words in a context that reverses or weakens their polarity. For example, row 6 says, "This app is no longer reliable on my phone." The word "reliable" is usually positive, but "no longer reliable" makes the full sentence negative. This tests whether the model understands the phrase as a whole or simply reacts to the positive trigger word.

## Refuted hypotheses

The domain_shift hypothesis was partly refuted. I expected the model to fail more often on sentences outside the app-review domain, such as business news, entertainment news, recipes, sports, and cybersecurity prose. However, the model achieved 80.0% accuracy in the `domain_shift` category. This suggests that the classifier does not always force app-review sentiment onto unrelated text. In several cases, it was able to treat factual sentences as neutral.

The length_extreme hypothesis was also partly refuted. I expected the model to struggle with very short examples like "Works.", "Broken again.", and "Okay.", or with long mixed sentences that contain many details. Instead, it achieved 80.0% accuracy in this category. This suggests that the model can handle both minimal and longer examples when the sentiment signal is clear enough.

The `other` category was also handled better than expected, with 80.0% accuracy. These examples included mixed or subtle sentiment cases. This result suggests that the model is not fragile on every difficult example. Its main weaknesses are more specific: sarcasm, negation, and misleading cue words.

## What the results reveal about the decision boundary

The adversarial results suggest that the model's decision boundary is strongly influenced by surface-level sentiment cues. When an example contains words like "great", "amazing", "perfect", "love", "reliable", or "beautiful", the model may lean positive unless the negative context is very explicit. Similarly, words like "broken", "crash", "terrible", and "unusable" can pull the model toward negative predictions.

The model appears less reliable when the correct label depends on compositional meaning. Negation and sarcasm require the model to understand how one part of the sentence changes another part. For example, "not bad" is not negative, and "Great, another update that broke everything" is not positive. These cases show that the classifier often responds more strongly to cue words than to the full intended meaning.

At the same time, the model did better than expected on `domain_shift` and `length_extreme` cases. This means the decision boundary is not only memorizing app-specific vocabulary. The model has learned some general sentiment patterns that transfer across domains and sentence lengths. However, it is still weak when the sentiment depends on irony, polarity reversal, or subtle context.

Overall, the adversarial evaluation shows that the classifier is acceptable for direct app-review sentiment but should not be trusted blindly on sarcastic, negated, or cue-word-conflict examples. A stronger production model would need additional targeted training data for sarcasm, negation, and mixed sentiment, plus confidence thresholds and human review for uncertain or linguistically complex cases.