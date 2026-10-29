create database if not exists flight_fare ; 


USE flight_fare;

CREATE TABLE flights (
    id INT AUTO_INCREMENT PRIMARY KEY,
    flight_date DATE,
    airline VARCHAR(30),
    flight_num VARCHAR(20),
    flight_class VARCHAR(20),
    source_city VARCHAR(30),
    dep_time TIME,
    dest_city VARCHAR(30),
    arr_time TIME,
    price DECIMAL(10,2),
    stops VARCHAR(10),
    duration_min DECIMAL(6,1)
);


SELECT COUNT(*) FROM flights;


SELECT COUNT(*) AS total_rows,
       SUM(flight_num IS NULL) AS null_flight_num,
       SUM(dep_time IS NULL) AS null_dep_time,
       SUM(arr_time IS NULL) AS null_arr_time
FROM flights;


SELECT flight_class,
       COUNT(*) AS total_flights,
       ROUND(AVG(price)) AS avg_price,
       MIN(price) AS min_price,
       MAX(price) AS max_price
FROM flights
GROUP BY flight_class;


WITH airline_avg AS (
    SELECT flight_class, airline,
           COUNT(*) AS flights,
           ROUND(AVG(price)) AS avg_price
    FROM flights
    WHERE airline NOT IN ('Unknown', 'starair', 'trujet')
    GROUP BY flight_class, airline
)
SELECT flight_class, airline, flights, avg_price,
       RANK() OVER (PARTITION BY flight_class ORDER BY avg_price DESC) AS price_rank
FROM airline_avg
ORDER BY flight_class, price_rank;



SELECT source_city, dest_city,
       ROUND(AVG(CASE WHEN flight_class = 'economy' THEN price END)) AS economy_avg,
       ROUND(AVG(CASE WHEN flight_class = 'business' THEN price END)) AS business_avg,
       ROUND(AVG(CASE WHEN flight_class = 'business' THEN price END) /
             AVG(CASE WHEN flight_class = 'economy' THEN price END), 1) AS business_x_times
FROM flights
GROUP BY source_city, dest_city
ORDER BY business_x_times DESC
LIMIT 10;


WITH route_avg AS (
    SELECT flight_class, source_city, dest_city,
           COUNT(*) AS flights,
           ROUND(AVG(price)) AS avg_price
    FROM flights
    GROUP BY flight_class, source_city, dest_city
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY flight_class ORDER BY avg_price DESC) AS rn
    FROM route_avg
)
SELECT flight_class, source_city, dest_city, flights, avg_price
FROM ranked
WHERE rn <= 3;



SELECT source_city, dest_city,
       COUNT(*) AS flights,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_all_flights
FROM flights
GROUP BY source_city, dest_city
ORDER BY flights DESC
LIMIT 5;


SELECT flight_class, stops,
       COUNT(*) AS flights,
       ROUND(AVG(price)) AS avg_price,
       ROUND(AVG(duration_min) / 60, 1) AS avg_hours
FROM flights
WHERE stops <> 'unknown'
GROUP BY flight_class, stops
ORDER BY flight_class, stops;


SELECT flight_class,
       CASE WHEN dep_time IS NULL THEN 'Unknown'
            WHEN HOUR(dep_time) BETWEEN 5 AND 11 THEN 'Morning'
            WHEN HOUR(dep_time) BETWEEN 12 AND 16 THEN 'Afternoon'
            WHEN HOUR(dep_time) BETWEEN 17 AND 20 THEN 'Evening'
            ELSE 'Night' END AS time_of_day,
       COUNT(*) AS flights,
       ROUND(AVG(price)) AS avg_price
FROM flights
GROUP BY flight_class, time_of_day
ORDER BY flight_class, avg_price DESC;



WITH weekly AS (
    SELECT YEARWEEK(flight_date, 1) AS yw,
           MIN(flight_date) AS week_start,
           flight_class,
           ROUND(AVG(price)) AS avg_price
    FROM flights
    GROUP BY YEARWEEK(flight_date, 1), flight_class
),
with_prev AS (
    SELECT *,
           LAG(avg_price) OVER (PARTITION BY flight_class ORDER BY yw) AS prev_week_price
    FROM weekly
)
SELECT week_start, flight_class, avg_price, prev_week_price,
       ROUND(100 * (avg_price - prev_week_price) / prev_week_price, 1) AS pct_change
FROM with_prev
ORDER BY flight_class, week_start;




SELECT quartile,
       COUNT(*) AS flights,
       MIN(price) AS min_price,
       MAX(price) AS max_price
FROM (
    SELECT price, NTILE(4) OVER (ORDER BY price) AS quartile
    FROM flights
    WHERE flight_class = 'economy'
) t
GROUP BY quartile
ORDER BY quartile;




WITH route_airline AS (
    SELECT source_city, dest_city, airline,
           COUNT(*) AS flights,
           AVG(price) AS airline_price
    FROM flights
    WHERE flight_class = 'economy'
      AND airline NOT IN ('Unknown', 'starair', 'trujet')
    GROUP BY source_city, dest_city, airline
),
route_avg AS (
    SELECT source_city, dest_city, AVG(price) AS route_price
    FROM flights
    WHERE flight_class = 'economy'
    GROUP BY source_city, dest_city
)

SELECT ra.source_city, ra.dest_city, ra.airline,
       ROUND(ra.airline_price) AS airline_price,
       ROUND(r.route_price) AS route_price,
       ROUND(100 * (ra.airline_price - r.route_price) / r.route_price, 1) AS pct_vs_route
FROM route_airline ra
JOIN route_avg r
  ON ra.source_city = r.source_city AND ra.dest_city = r.dest_city
WHERE ra.flights >= 100
ORDER BY pct_vs_route DESC
LIMIT 10;




CREATE INDEX idx_route ON flights (source_city, dest_city);

CREATE VIEW route_summary AS
SELECT source_city, dest_city, flight_class,
       COUNT(*) AS flights,
       ROUND(AVG(price)) AS avg_price,
       ROUND(AVG(duration_min)) AS avg_duration_min
FROM flights
GROUP BY source_city, dest_city, flight_class;







