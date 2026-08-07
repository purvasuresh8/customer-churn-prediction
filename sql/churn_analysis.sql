SELECT
    Churn,
    COUNT(*) AS CustomerCount
FROM customers
GROUP BY Churn;
