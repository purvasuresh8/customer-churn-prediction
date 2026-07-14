import os
import shap
import pandas as pd
import matplotlib.pyplot as plt


class ModelExplainer:
    """
    Generate model explainability artifacts using SHAP.
    """

    def __init__(
        self,
        model,
        X_train,
        X_test,
        feature_names=None
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

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

        shap_values = explainer(
            self.X_test
        )

        return shap_values

    def generate_summary_plot(self):
        """
        Generate SHAP summary plot.
        """

        shap_values = (
            self.calculate_shap_values()
        )

        plt.figure()

        shap.summary_plot(
            shap_values,
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
            "SHAP Summary Plot saved."
        )

    def generate_bar_plot(self):
        """
        Generate feature importance bar plot.
        """

        shap_values = (
            self.calculate_shap_values()
        )

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
            "SHAP Feature Importance Plot saved."
        )

    def generate_waterfall_plot(
        self,
        customer_index=0
    ):
        """
        Explain a single prediction.
        """

        shap_values = (
            self.calculate_shap_values()
        )

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
            "SHAP Waterfall Plot saved."
        )

    def generate_dependence_plot(
        self,
        feature_name
    ):
        """
        Show impact of one feature.
        """

        shap_values = (
            self.calculate_shap_values()
        )

        shap.dependence_plot(
            feature_name,
            shap_values.values,
            self.X_test,
            show=False
        )

        plt.savefig(
            f"visualizations/shap/{feature_name}_dependence.png",
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"{feature_name} dependence plot saved."
        )

    def generate_all_reports(self):
        """
        Generate all explainability outputs.
        """

        self.generate_summary_plot()

        self.generate_bar_plot()

        self.generate_waterfall_plot()

        print(
            "All SHAP reports generated."
        )


def calculate_top_churn_drivers(
    model,
    X_train,
    X_test
):
    """
    Return most important features.
    """

    explainer = shap.Explainer(
        model,
        X_train
    )

    shap_values = explainer(X_test)

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance":
            abs(shap_values.values).mean(axis=0)
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    return importance_df
