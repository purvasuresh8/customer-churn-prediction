import pandas as pd


class DataValidator:
    """
    Data quality validation for customer churn dataset.
    """

    def __init__(self, df):
        self.df = df
        self.validation_report = {}

    def check_missing_values(self):
        """
        Check missing values by column.
        """

        missing = self.df.isnull().sum()

        self.validation_report["missing_values"] = (
            missing[missing > 0].to_dict()
        )

        return missing

    def check_duplicate_records(self):
        """
        Check duplicate rows.
        """

        duplicates = self.df.duplicated().sum()

        self.validation_report["duplicate_rows"] = int(
            duplicates
        )

        return duplicates

    def check_empty_strings(self):
        """
        Check empty string values.
        """

        empty_counts = {}

        for column in self.df.columns:

            count = (
                self.df[column]
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

            if count > 0:
                empty_counts[column] = int(count)

        self.validation_report["empty_strings"] = (
            empty_counts
        )

        return empty_counts

    def validate_tenure(self):
        """
        Validate tenure values.
        """

        if "tenure" not in self.df.columns:
            return 0

        invalid_tenure = (
            self.df["tenure"] < 0
        ).sum()

        self.validation_report[
            "invalid_tenure_records"
        ] = int(invalid_tenure)

        return invalid_tenure

    def validate_monthly_charges(self):
        """
        Check for negative monthly charges.
        """

        if "MonthlyCharges" not in self.df.columns:
            return 0

        invalid_charges = (
            self.df["MonthlyCharges"] < 0
        ).sum()

        self.validation_report[
            "invalid_monthly_charges"
        ] = int(invalid_charges)

        return invalid_charges

    def validate_total_charges(self):
        """
        Check TotalCharges values.
        """

        if "TotalCharges" not in self.df.columns:
            return 0

        charges = pd.to_numeric(
            self.df["TotalCharges"],
            errors="coerce"
        )

        invalid_total = (
            charges < 0
        ).sum()

        self.validation_report[
            "invalid_total_charges"
        ] = int(invalid_total)

        return invalid_total

    def validate_target_column(self):
        """
        Check Churn column values.
        """

        if "Churn" not in self.df.columns:
            return []

        valid_values = [
            "Yes",
            "No",
            0,
            1
        ]

        invalid_rows = self.df[
            ~self.df["Churn"].isin(valid_values)
        ]

        self.validation_report[
            "invalid_churn_values"
        ] = len(invalid_rows)

        return invalid_rows

    def validate_customer_id(self):
        """
        Check uniqueness of customer IDs.
        """

        if "customerID" not in self.df.columns:
            return 0

        duplicate_ids = self.df[
            "customerID"
        ].duplicated().sum()

        self.validation_report[
            "duplicate_customer_ids"
        ] = int(duplicate_ids)

        return duplicate_ids

    def generate_report(self):
        """
        Run all validations.
        """

        self.check_missing_values()

        self.check_duplicate_records()

        self.check_empty_strings()

        self.validate_tenure()

        self.validate_monthly_charges()

        self.validate_total_charges()

        self.validate_target_column()

        self.validate_customer_id()

        return self.validation_report

    def print_report(self):
        """
        Print validation report.
        """

        report = self.generate_report()

        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        for key, value in report.items():
            print(f"{key}: {value}")

        print("=" * 60)

        return report
      
    def validate_dataset(df):
    """
    Quick validation function.
    """

    validator = DataValidator(df)

    return validator.print_report()
