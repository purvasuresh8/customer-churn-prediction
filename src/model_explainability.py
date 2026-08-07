import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class ModelExplainer:
    """
    Generate model explainability artifacts using SHAP.
    """

    def __init__(
        self,
        model,
        X_train,
        X_test
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test

        os.makedirs(
            "visualizations/shap",
            exist_ok=True
        )

    def get_explainer(self):
        """
        Create SHAP explainer.
        """
        return shap.Explainer(
            self.model,
            self.X_train
        )

    def calculate_shap_values(self):
        """
        Compute SHAP values.
        """
        explainer = self.get_explainer()

        return explainer(
            self.X_test
        )

    def generate_summary_plot(self):
        """
        Generate SHAP summary plot.
        """

        shap_values = self.calculate_shap_values()

        plt.figure()

        shap.summary_plot(
            shap_values.values,
            self.X_test,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            "visualizations/shap/shap_summary.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            "✅ SHAP summary plot saved."
        )

    def generate_bar_plot(self):
        """
        Generate SHAP feature importance plot.
        """

        shap_values = self.calculate_shap_values()

        plt.figure()

        shap.plots.bar(
            shap_values,
            show=False
        )

        plt.savefig(
            "visualizations/shap/shap_feature_importance.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            "✅ SHAP feature importance plot saved."
        )

    def generate_waterfall_plot(
        self,
        customer_index=0
    ):
        """
        Explain a single prediction.
        """

        shap_values = self.calculate_shap_values()

        plt.figure()

        shap.plots.waterfall(
            shap_values[customer_index],
            show=False
        )

        plt.savefig(
            f"visualizations/shap/customer_{customer_index}_waterfall.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            "✅ SHAP waterfall plot saved."
        )

    def generate_all_reports(self):
        """
        Generate all explainability reports.
        """

        self.generate_summary_plot()

        self.generate_bar_plot()

        self.generate_waterfall_plot()

        print(
            "\n✅ All SHAP reports generated."
        )


def calculate_top_churn_drivers(
    model,
    X_train,
    X_test
):
    """
    Calculate feature importance using SHAP values.
    """

    explainer = shap.Explainer(
        model,
        X_train
    )

    shap_values = explainer(
        X_test
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": (abs(shap_values.values)
        .mean(axis=0)
        )
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    return importance_df


def save_top_drivers(
    model,
    X_train,
    X_test
):
    """
    Save top churn drivers report.
    """

    os.makedirs(
        "reports",
        exist_ok=True
    )

    top_drivers = calculate_top_churn_drivers(
        model,
        X_train,
        X_test
    )

    top_drivers.to_csv(
        "reports/top_churn_drivers.csv",
        index=False
    )

    print(
        "\n✅ Top churn drivers saved."
    )

    print(
        top_drivers.head(10)
    )


if __name__ == "__main__":

    print("Loading dataset...")

    df = pd.read_csv(
        "data/processed/feature_engineered_data.csv"
    )

    X = df.drop(
        "Churn",
        axis=1
    )
    X = X.astype(float)

    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Loading model...")

    model = joblib.load(
        "models/logistic_regression.pkl"
    )

    explainer = ModelExplainer(
        model=model,
        X_train=X_train,
        X_test=X_test
    )

    explainer.generate_all_reports()

    save_top_drivers(
        model,
        X_train,
        X_test
    )

    print(
        "\n✅ Explainability pipeline completed."
    )
    