SET ANSI_WARNINGS OFF;

---- 1) load cohort1 (total) ----
IF OBJECT_ID('tempdb..#temp_cohort_total') IS NOT NULL
	DROP TABLE #temp_cohort_total

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO #temp_cohort_total
FROM @target_database_schema.@target_person_table_total
GROUP BY person_id

---- 2) load cohort2 (case) ---

IF OBJECT_ID('tempdb..#temp_cohort_case') IS NOT NULL
	DROP TABLE #temp_cohort_case

SELECT person_id AS subject_id, min(cohort_start_date) AS cohort_start_date
INTO #temp_cohort_case
FROM @target_database_schema.@target_person_table_case
GROUP BY person_id

---- 3) total + case cohort ---

IF OBJECT_ID('tempdb..#temp_cohort') IS NOT NULL
	DROP TABLE #temp_cohort

SELECT t.subject_id
, t.cohort_start_date as cohort_start_date1
, c.cohort_start_date as cohort_start_date2
, ISNULL(c.cohort_start_date, t.cohort_start_date) AS cohort_start_date 
INTO #temp_cohort
FROM (SELECT * FROM #temp_cohort_total) t
LEFT JOIN (SELECT * FROM #temp_cohort_case) c
ON t.subject_id = c.subject_id

--SELECT * FROM #temp_cohort

---- 4) 3 times sampling ----

IF OBJECT_ID('tempdb..#temp_cohort_sampling') IS NOT NULL
	DROP TABLE #temp_cohort_sampling
	
SELECT TOP(3*(SELECT COUNT(*) FROM #temp_cohort WHERE cohort_start_date2 is not null)) *
INTO #temp_cohort_sampling
FROM #temp_cohort
ORDER BY cohort_start_date2 DESC

--select * from #temp_cohort_sampling

---- 5) join table (cohort & person) ----

IF OBJECT_ID('tempdb..#temp_person') IS NOT NULL
	DROP TABLE #temp_person

SELECT * 
INTO #temp_person
FROM #temp_cohort_sampling
LEFT JOIN @cdm_database_schema.person
ON #temp_cohort_sampling.subject_id = @cdm_database_schema.person.person_id

---- 6) save table (person table) ----

IF OBJECT_ID('@target_database_schema.@target_person_table') IS NOT NULL
	DROP TABLE @target_database_schema.@target_person_table

SELECT person_id, cohort_start_date, (YEAR(cohort_start_date)-year_of_birth) AS age, gender_source_value
INTO @target_database_schema.@target_person_table
FROM #temp_person

-- SELECT * FROM @target_database_schema.@target_person_table
