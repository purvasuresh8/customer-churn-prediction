SELECT
    Contract,
    AVG(Churn) AS churn_rate
FROM customers
GROUP BY Contract
ORDER BY churn_rate DESC;
