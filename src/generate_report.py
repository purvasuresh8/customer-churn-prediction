from pathlib import Path
import pandas as pd
from datetime import datetime


class ReportGenerator:

    def __init__(
        self,
        dataset,
        model_metrics,
        prediction_data=None
    ):
        self.dataset = dataset
        self.model_metrics = model_metrics
        self.prediction_data = prediction_data

    def dataset_summary(self):
        """
        Generate dataset statistics.
        """

        summary = {
            "Total Records": len(self.dataset),
            "Total Features": len(self.dataset.columns),
            "Missing Values":
                int(self.dataset.isnull().sum().sum())
        }

        if "Churn" in self.dataset.columns:
            summary["Churn Rate (%)"] = round(
                self.dataset["Churn"].mean() * 100,
                2
            )

        return summary

    def model_summary(self):
        """
        Summarize model performance.
        """

        return self.model_metrics

    def top_retention_recommendations(self):
        """
        Count recommendation occurrences.
        """

        if self.prediction_data is None:
            return {}

        if "Recommendation" not in self.prediction_data.columns:
            return {}

        return (
            self.prediction_data["Recommendation"]
            .value_counts()
            .to_dict()
        )

    def generate_business_insights(self):
        """
        Generate business insights.
        """

        insights = []

        if "Churn" in self.dataset.columns:

            churn_rate = (
                self.dataset["Churn"].mean() * 100
            )

            insights.append(
                f"Overall churn rate is {churn_rate:.2f}%."
            )

        if (
            self.prediction_data is not None and
            "RiskLevel" in self.prediction_data.columns
        ):

            high_risk = (
                self.prediction_data["RiskLevel"]
                .eq("High")
                .sum()
            )

            insights.append(
                f"{high_risk} customers are classified "
                f"as high-risk."
            )

        return insights

    def generate_report_text(self):
        """
        Generate complete report.
        """

        report = []

        report.append("=" * 70)
        report.append(
            "CUSTOMER CHURN ANALYTICS REPORT"
        )
        report.append("=" * 70)

        report.append(
            f"Generated: {datetime.now()}"
        )

        report.append("\n")

        report.append("DATASET SUMMARY")
        report.append("-" * 30)

        for key, value in self.dataset_summary().items():
            report.append(
                f"{key}: {value}"
            )

        report.append("\n")

        report.append("MODEL PERFORMANCE")
        report.append("-" * 30)

        for key, value in self.model_summary().items():
            report.append(
                f"{key}: {value}"
            )

        report.append("\n")

        report.append(
            "BUSINESS INSIGHTS"
        )
        report.append("-" * 30)

        for insight in self.generate_business_insights():
            report.append(
                f"• {insight}"
            )

        report.append("\n")

        report.append(
            "RETENTION RECOMMENDATIONS"
        )
        report.append("-" * 30)

        recommendations = (
            self.top_retention_recommendations()
        )

        if recommendations:

            for action, count in recommendations.items():

                report.append(
                    f"• {action} ({count} customers)"
                )

        return "\n".join(report)

    def save_report(
        self,
        output_file
    ):
        """
        Save report to file.
        """

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        report_text = (
            self.generate_report_text()
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(report_text)

        print(
            f"Report saved: {output_file}"
        )


def generate_reports(
    dataset,
    metrics,
    prediction_data=None
):
    """
    Generate all project reports.
    """

    generator = ReportGenerator(
        dataset=dataset,
        model_metrics=metrics,
        prediction_data=prediction_data
    )

    generator.save_report(
        "reports/churn_insights_report.txt"
    )
