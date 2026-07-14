import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate classification model performance.
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, probabilities)
    }

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 50}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return metrics


def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Generate and save confusion matrix.
    """

    os.makedirs("visualizations", exist_ok=True)

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        f"visualizations/{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.close()


def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name
):
    """
    Generate and save ROC curve.
    """

    os.makedirs("visualizations", exist_ok=True)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{model_name} ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"visualizations/{model_name.lower().replace(' ', '_')}_roc_curve.png"
    )

    plt.close()


def save_feature_importance(
    model,
    feature_names,
    model_name
):
    """
    Save Random Forest feature importance.
    """

    if not hasattr(model, "feature_importances_"):
        return

    os.makedirs("reports", exist_ok=True)

    import pandas as pd

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    feature_df.to_csv(
        f"reports/{model_name.lower().replace(' ', '_')}_feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    top_features = feature_df.head(15)

    sns.barplot(
        data=top_features,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"{model_name} Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        f"visualizations/{model_name.lower().replace(' ', '_')}_feature_importance.png"
    )

    plt.close()


def evaluate_and_visualize(
    model,
    X_test,
    y_test,
    model_name,
    feature_names=None
):
    """
    Complete evaluation workflow.
    """

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        model_name
    )

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
        model_name
    )

    plot_roc_curve(
        model,
        X_test,
        y_test,
        model_name
    )

    if feature_names:
        save_feature_importance(
            model,
            feature_names,
            model_name
        )

    return metrics
