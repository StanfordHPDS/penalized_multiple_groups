CREATE OR REPLACE TABLE som-nero-phi-sherrir-afc.cnaks_afc.cohort_fp AS (
-- Pull in ESRD concepts
    WITH esrd_concepts AS (
    SELECT *
    FROM `som-nero-phi-sherrir-afc.afc0524phiomop.concept` AS a
    WHERE (`concept_name` LIKE "%nd stage renal disease%") OR (`concept_name` LIKE "%nd stage renal failure%")),

-- Get ESRD records from the concepts
    ESRD_records AS  (
      SELECT 
        a.observation_id as occurrence_id,
        a.person_id,
        a.observation_concept_id as concept_id,
        a.load_table_id,
        a.observation_date as date,
        a.provider_id,
        b.concept_name,
        b.vocabulary_id
    FROM `som-nero-phi-sherrir-afc.afc0524phiomop.observation` AS a
    INNER JOIN esrd_concepts AS b
    ON a.observation_concept_id=b.concept_id
    UNION ALL
    SELECT 
        a.condition_occurrence_id as occurrence_id,
        a.person_id,
        a.condition_concept_id as concept_id,
        a.load_table_id,
        a.condition_start_date as date,
        a.provider_id,
        b.concept_name,
        b.vocabulary_id
    FROM `som-nero-phi-sherrir-afc.afc0524phiomop.condition_occurrence` AS a
    INNER JOIN esrd_concepts AS b
    ON a.condition_concept_id=b.concept_id or a.condition_type_concept_id =b.concept_id
    or a.condition_source_concept_id = b.concept_id),

-- find patients with ESRD record or low eGFR
    unique_patients AS (
        SELECT distinct(person_id), min(date) as code_date, max(date) as max_code_date, 1 as code
        FROM ESRD_records
        GROUP BY person_id
    ), 

    egfr_dx_date AS ( 
        SELECT distinct(person_id), min(measurement_date) as first_g5_test_date, max(measurement_date) as last_g5_test_date
        FROM `som-nero-phi-sherrir-afc.agataf_omop.eGFR_gen_incl_noacute`
        WHERE egfr < 15
        GROUP BY person_id
    ), 

    lowest_egfr AS (
            SELECT a.person_id, a.egfr as lowest_egfr, min(a.egfr) < 15 as g5, min(a.measurement_date) as lowest_egfr_test_date
            FROM `som-nero-phi-sherrir-afc.agataf_omop.eGFR_gen_incl_noacute` a
            INNER JOIN (
                SELECT person_id, min(egfr) as egfr
                FROM `som-nero-phi-sherrir-afc.agataf_omop.eGFR_gen_incl_noacute`
                WHERE egfr IS NOT NULL
                GROUP BY person_id
            ) b ON a.person_id = b.person_id AND a.egfr = b.egfr
            GROUP BY a.person_id, a.egfr
    ),

    egfr_summary AS (
        SELECT a.person_id, lowest_egfr, g5, a.lowest_egfr_test_date,
            b.first_g5_test_date, b.last_g5_test_date
        FROM lowest_egfr AS a
        LEFT OUTER JOIN egfr_dx_date AS b
        ON a.person_id = b.person_id
    ),

    semifinal as (
        SELECT a.*, b.lowest_egfr, b.g5, b.person_id as person_id_b, b.lowest_egfr_test_date, 
            b.first_g5_test_date, b.last_g5_test_date, (b.g5 and (a.code = 1)) as concordant, 
            Case When first_g5_test_date is null Then code_date
                When code_date is null Then first_g5_test_date 
                When first_g5_test_date < code_date Then first_g5_test_date
                Else code_date
                End As dx_date, 
            Case When last_g5_test_date is null Then max_code_date
                When max_code_date is null Then last_g5_test_date 
                When last_g5_test_date < max_code_date Then last_g5_test_date
                Else max_code_date
                End As last_dx_date, 
            Case When lowest_egfr_test_date is null Then null
                When g5 Then first_g5_test_date 
                Else lowest_egfr_test_date
                End As critical_test_date
        FROM unique_patients AS a
        FULL OUTER JOIN egfr_summary AS b
        ON a.person_id=b.person_id
        WHERE (b.lowest_egfr < 15) OR (a.code = 1)
    ),

    ESRD_patients as (SELECT CASE WHEN person_id IS NOT NULL THEN person_id ELSE person_id_b END as person_id, 
            code_date, code, lowest_egfr, g5, lowest_egfr_test_date, first_g5_test_date, last_g5_test_date, 
            concordant, dx_date, last_dx_date, date_diff(code_date, critical_test_date, day) as code_test_dt
        FROM semifinal   
    ),

-- Bring in measurements: BP, bili, HA1c
    measurements AS (
        SELECT measurement_id, person_id, measurement_concept_id, measurement_date, 
              CASE WHEN measurement_concept_id in (4184637, 36304734) and value_as_number > 50
                        THEN (value_as_number + 46.7) / 28.7
                ELSE value_as_number
                END
                AS value_as_number,
            unit_concept_id, measurement_source_value
        FROM `som-nero-phi-sherrir-afc.afc0524phiomop.measurement`
        WHERE measurement_concept_id in (4298393, 3012888, 4152194, 3004249, 4184637, 36304734, 4118986, 3024148, 4216632, 3024128) and 
            value_as_number is not null and value_as_number > 0 and value_as_number < 300 and
            EXTRACT(YEAR FROM measurement_date) in (2017,2018)
    ), 

    pre_cohort AS (
        SELECT CASE WHEN a.person_id is not null THEN a.person_id ELSE b.person_id END AS person_id, 
            EXTRACT(YEAR FROM max(b.dx_date)) < 2020 as ESRD, max(b.dx_date) as dx_date, max(a.measurement_date) as last_measure_date
        FROM measurements a full outer join ESRD_patients b
        ON a.person_id = b.person_id
        WHERE measurement_concept_id in (4298393, 3012888, 4152194, 3004249)
        GROUP BY person_id
    ), 
-- Bring in eGFR
    egfrs AS ( 
        SELECT distinct(person_id), min(egfr) as min_egfr, max(measurement_date) as last_egfr_date
        FROM `som-nero-phi-sherrir-afc.agataf_omop.eGFR_gen_incl_noacute`
        WHERE EXTRACT(YEAR FROM measurement_date) in (2017,2018)
        GROUP BY person_id
    ), 

    precohort AS (
        SELECT a.*, min_egfr, last_egfr_date
        FROM pre_cohort a full outer join egfrs b
        ON a.person_id = b.person_id
        where not IFNULL(esrd,FALSE) or EXTRACT(YEAR FROM dx_date) > 2016
    ),
-- integrate measures as cohort analytic
    cohort AS (SELECT a.person_id, max(a.ESRD) as ESRD, max(a.dx_date) as dx_date, min(min_egfr) as min_egfr, max(last_egfr_date) as last_egfr_date, 
            max(case when measurement_concept_id in (4298393, 3012888) then value_as_number end) as max_diastolic, 
            avg(case when measurement_concept_id in (4298393, 3012888) and EXTRACT(YEAR FROM b.measurement_date) > 2016 then value_as_number end) 
                as mean_diastolic, 
            max(case when measurement_concept_id in (4152194, 3004249) then value_as_number end) as max_systolic, 
            avg(case when measurement_concept_id in (4152194, 3004249) and EXTRACT(YEAR FROM b.measurement_date) > 2016 then value_as_number end) 
                as mean_systolic, 
            max(case when measurement_concept_id in (4184637, 36304734) then value_as_number end) as max_ha1c, 
            avg(case when measurement_concept_id in (4184637, 36304734) and EXTRACT(YEAR FROM b.measurement_date) > 2016 then value_as_number end) as mean_ha1c, 
            max(case when measurement_concept_id in (4118986, 3024148, 4216632, 3024128) then value_as_number end) as max_bili, 
            avg(case when measurement_concept_id in (4118986, 3024148, 4216632, 3024128) and EXTRACT(YEAR FROM b.measurement_date) > 2016 then value_as_number end) as mean_bili 
        FROM precohort a, measurements b
        WHERE a.person_id = b.person_id
        GROUP BY a.person_id
    ),

-- Bring in diagnoses: get codes and look them up
    spec_concepts AS (
        SELECT concept_id, (`concept_name` LIKE "%glomerulonephritis%") as glomerulonephritis, 
            (`concept_name` LIKE "%proteinuria%") as proteinuria, (`concept_name` LIKE "%pyelonephritis%") as pyelonephritis
        FROM `som-nero-phi-sherrir-afc.afc0524phiomop.concept` AS a
        WHERE ((`concept_name` LIKE "%glomerulonephritis%") OR (`concept_name` LIKE "%proteinuria%") OR
        (`concept_name` LIKE "%pyelonephritis%")) AND (`domain_id` IN ('Condition')) 
    ),
    
    dx AS (
        SELECT person_id, condition_start_date, 
            condition_concept_id in (44784621,443611,443601,44782690,43021852,45768812,44782728,45763855,45763854,43531578,45768812,443612,44782429,43531653,443614,43021852,443597) as ckd,
            condition_concept_id in (43020424, 45768449, 37200492,37200493,37200494) as htn_emerg,
            condition_concept_id in (44809548, 37208172) or SUBSTRING(condition_source_value,1,3) = "E11" as dm2,
            condition_concept_id in (201620, 45548655,44822046) as kidstone,
            condition_concept_id in (45612141, 45592125,45567838, 36713449,4220439,433813) as obstruction,
            glomerulonephritis, proteinuria, pyelonephritis
        FROM `som-nero-phi-sherrir-afc.afc0524phiomop.condition_occurrence` a left outer join spec_concepts b
        ON a.condition_concept_id = b.concept_id
        WHERE EXTRACT(YEAR FROM condition_start_date) in (2017,2018)
    ), 
-- Bring in demographics
    people_merged AS (
        SELECT a.*, CASE WHEN b.gender_concept_id = 0 THEN NULL ELSE b.gender_concept_id = 8532 END AS female,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE b.race_concept_id in (8527,38003614) END AS white,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE b.race_concept_id in (8516,38003597,38003598,38003599,38003600) END AS black,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE (b.race_concept_id = 8515 or (b.race_concept_id >= 38003574 and b.race_concept_id <= 38003596)) END AS asian,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE b.race_concept_id in (8657,38003572,38003573) END AS AIAN,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE b.race_concept_id in (8557,38003610,38003611,38003612,38003613) END AS NHPI,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE b.race_concept_id in (38003615,38003616) END AS MENA,
        CASE WHEN b.race_concept_id = 0 THEN NULL ELSE (b.race_concept_id >= 38003601 and b.race_concept_id <= 38003609) END AS LatAm,
        CASE WHEN b.ethnicity_concept_id = 0 THEN NULL ELSE b.ethnicity_concept_id = 38003563 END AS hispanic,
        2019 - EXTRACT(YEAR FROM birth_datetime) AS age
        FROM cohort a, `som-nero-phi-sherrir-afc.afc0524phiomop.person` b
        WHERE a.person_id = b.person_id AND max_diastolic is not null AND max_systolic is not null
    ),
-- Collapse demos, dx to binaries and merge into cohort
    dx_sum AS (
        SELECT a.person_id, max(b.ckd) as ckd, count(case when b.ckd THEN b.ckd ELSE NULL END) > 1 as ckd2, 
            max(b.htn_emerg) as htn_emerg, max(b.dm2) as dm2, max(b.kidstone) as kidstone, max(b.obstruction) as obstruction, 
            max(b.glomerulonephritis) as glomerulonephritis, max(b.proteinuria) as proteinuria, max(b.pyelonephritis) as pyelonephritis 
        FROM people_merged a, dx b
        WHERE a.person_id = b.person_id
        GROUP BY a.person_id
    ),

    dx_cohort AS (SELECT a.*, b.ckd, b.ckd2, b.htn_emerg, b.dm2, b.kidstone, b.obstruction, 
            b.glomerulonephritis, b.proteinuria, b.pyelonephritis 
    FROM people_merged a left outer join dx_sum b
    ON a.person_id = b.person_id
    ),

-- Keep only practices with at least one ESRD patient 2017-9

    people_merged2 AS (
        SELECT a.*, b.provider_id, b.care_site_id
        FROM dx_cohort a, `som-nero-phi-sherrir-afc.afc0524phiomop.person` b
        WHERE a.person_id = b.person_id
    ),

    esrd_practices AS (
        SELECT DISTINCT care_site_id
        FROM people_merged2
        WHERE ESRD = TRUE and EXTRACT(YEAR FROM dx_date) in (2017,2018,2019)
    )

    SELECT a.*
    FROM people_merged2 a, esrd_practices b
    WHERE a.care_site_id = b.care_site_id
)
