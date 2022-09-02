---- 1) load cohort1 (total) ----

DROP TABLE IF EXISTS temp_cohort_total;

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO TEMP temp_cohort_total
FROM @target_database_schema.@target_person_table_total
GROUP BY person_id;

---- 2) load cohort2 (case) ---

DROP TABLE IF EXISTS temp_cohort_case;

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO TEMP temp_cohort_case
FROM @target_database_schema.@target_person_table_case
GROUP BY person_id;

---- 3) total + case cohort ---

DROP TABLE IF EXISTS temp_cohort;

SELECT t.subject_id
, t.cohort_start_date as cohort_start_date1
, c.cohort_start_date as cohort_start_date2
, COALESCE(c.cohort_start_date, t.cohort_start_date) AS cohort_start_date 
INTO TEMP temp_cohort
FROM (SELECT * FROM temp_cohort_total) t
LEFT JOIN (SELECT * FROM temp_cohort_case) c
ON t.subject_id = c.subject_id;

--SELECT * FROM temp_cohort

---- 4) 3 times sampling ----

DROP TABLE IF EXISTS temp_cohort_sampling;

SELECT *
INTO TEMP temp_cohort_sampling
FROM temp_cohort
ORDER BY cohort_start_date2 DESC
LIMIT (3*(SELECT COUNT(*) FROM temp_cohort WHERE cohort_start_date2 is not null));

-- select * from temp_cohort_sampling

---- 5) join table (cohort & person) ----

DROP TABLE IF EXISTS temp_person;

SELECT * 
INTO TEMP temp_person
FROM temp_cohort_sampling
LEFT JOIN @cdm_database_schema.person
ON temp_cohort_sampling.subject_id = @cdm_database_schema.person.person_id;

---- 6) save table (person table) ----

DROP TABLE IF EXISTS @target_database_schema.@target_person_table;

SELECT person_id, cohort_start_date, gender_concept_id, (date_part('year', cohort_start_date)-year_of_birth) as age
INTO @target_database_schema.@target_person_table
FROM temp_person;

-- SELECT * FROM @target_database_schema.@target_person_table
