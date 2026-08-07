SELECT
    gender,
    COUNT(*) AS customers
FROM customers
GROUP BY gender;