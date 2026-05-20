"""
Stretch Thursday — Adversarial Evaluation.

Load a fine-tuned classifier, run it against adversarial_set.csv, and write
results.csv. Read label names from model.config.id2label — do not hard-code.
"""

import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt



def load_model(model_path: str = "model"):
    """
    Load model and tokenizer from a local path or HF Hub id.

    Defaults to local 'model' (your Lab 7A checkpoint). CI overrides via MODEL_PATH env.
    """
    # TODO: AutoModelForSequenceClassification.from_pretrained(model_path)
    # TODO: AutoTokenizer.from_pretrained(model_path)
    # TODO: return both
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.eval()

    return model, tokenizer


def run_against_set(adv_csv_path: str, model, tokenizer) -> pd.DataFrame:
    """
    Run the model on every row of adv_csv_path. Return a DataFrame with all
    original columns plus predicted_label, predicted_probability, correct.

    Read label names from model.config.id2label — do not hard-code class names.
    """
    # TODO: read adv_csv_path with pandas
    # TODO: for each row, tokenize + forward pass + softmax + argmax
    # TODO: convert argmax index to label name via model.config.id2label
    # TODO: build a results DataFrame with predicted_label, predicted_probability, correct
    # TODO: return the DataFrame
    df = pd.read_csv(adv_csv_path)

    predicted_labels = []
    predicted_probabilities = []
    correct_values = []

    for _, row in df.iterrows():
        text = row["text"]
        expected_label = row["expected_label"]

        inputs = tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)

        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0, pred_idx].item()

        label_name = model.config.id2label[pred_idx]

        predicted_labels.append(label_name)
        predicted_probabilities.append(float(pred_prob))
        correct_values.append(label_name == expected_label)

    results_df = df.copy()
    results_df["predicted_label"] = predicted_labels
    results_df["predicted_probability"] = predicted_probabilities
    results_df["correct"] = correct_values

    return results_df


def calculate_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate overall accuracy and per-category accuracy.

    Returns a summary DataFrame with:
    hypothesis_category, total, correct, incorrect, accuracy
    """
    summary = (
        results_df
        .groupby("hypothesis_category")["correct"]
        .agg(total="count", correct="sum")
        .reset_index()
    )

    summary["incorrect"] = summary["total"] - summary["correct"]
    summary["accuracy"] = summary["correct"] / summary["total"]

    overall_row = pd.DataFrame([{
        "hypothesis_category": "overall",
        "total": len(results_df),
        "correct": results_df["correct"].sum(),
        "incorrect": len(results_df) - results_df["correct"].sum(),
        "accuracy": results_df["correct"].mean()
    }])

    summary = pd.concat([summary, overall_row], ignore_index=True)

    return summary


def plot_accuracy(summary_df: pd.DataFrame, output_path: str = "accuracy_by_category.png") -> None:
    """
    Plot accuracy by hypothesis category and save it as a PNG file.
    """
    plot_df = summary_df[summary_df["hypothesis_category"] != "overall"]

    plt.figure(figsize=(10, 6))
    plt.bar(
        plot_df["hypothesis_category"],
        plot_df["accuracy"]
    )

    plt.title("Adversarial Accuracy by Hypothesis Category")
    plt.xlabel("Hypothesis Category")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.xticks(rotation=30, ha="right")

    for i, value in enumerate(plot_df["accuracy"]):
        plt.text(
            i,
            value + 0.02,
            f"{value:.0%}",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correct_incorrect(summary_df: pd.DataFrame, output_path: str = "correct_vs_incorrect.png") -> None:
    """
    Plot correct vs incorrect counts by category and save it as a PNG file.
    """
    plot_df = summary_df[summary_df["hypothesis_category"] != "overall"]

    x = range(len(plot_df))

    plt.figure(figsize=(10, 6))
    plt.bar(
        x,
        plot_df["correct"],
        label="Correct"
    )
    plt.bar(
        x,
        plot_df["incorrect"],
        bottom=plot_df["correct"],
        label="Incorrect"
    )

    plt.title("Correct vs Incorrect Predictions by Category")
    plt.xlabel("Hypothesis Category")
    plt.ylabel("Number of Examples")
    plt.xticks(x, plot_df["hypothesis_category"], rotation=30, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def main() -> None:
    """Orchestrate; write results.csv."""
    model_path = os.environ.get("MODEL_PATH", "model")
    adv_csv = os.environ.get("ADVERSARIAL_CSV", "adversarial_set.csv")
    out_csv = os.environ.get("RESULTS_CSV", "results.csv")

    model, tokenizer = load_model(model_path)
    df = run_against_set(adv_csv, model, tokenizer)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")
    summary_df = calculate_accuracy(df)
    summary_df.to_csv("accuracy_summary.csv", index=False)

    print("\nAccuracy summary:")
    print(summary_df)

    plot_accuracy(summary_df)
    plot_correct_incorrect(summary_df)

    print("Wrote accuracy_summary.csv")
    print("Wrote accuracy_by_category.png")
    print("Wrote correct_vs_incorrect.png")


if __name__ == "__main__":
    main()