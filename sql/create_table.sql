DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    gender INTEGER,
    SeniorCitizen INTEGER,
    Partner INTEGER,
    Dependents INTEGER,
    tenure REAL,
    PhoneService INTEGER,
    PaperlessBilling INTEGER,
    MonthlyCharges REAL,
    TotalCharges REAL,
    Churn INTEGER,
    AvgMonthlySpend REAL,
    HighValueCustomer INTEGER,
    ServiceCount INTEGER,
    EngagementScore REAL
);
