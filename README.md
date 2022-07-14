# CDM기반 의약품 부작용 예측모델 (간독성 / 신독성)

해당 프로젝트는 공통데이터모델(CDM)기반 특정 약물 복용군 대상 간독성/신독성 부작용 예측모델을 생성하는 것을 목표로 한다. 

## Description

* 0_cohort 
   - sql문을 실행시켜 DB에 person_{drug} / person_{meas} 생성 

* 1_importsql
   - DB로 부터 데이터를 읽어와 첫 부작용 발생일 추가 및 데이터를 파일로 저장

* 2_feature_selection
   - 모든 concept_set에서 유의미한 Feature 선정

* 3_preprocessing_lstm
   - TimeSeries data 형태로 데이터 전처리 
    (Pivotting / feature selection / imputation.. / window sliding)
   - Feature 분포 확인 

* 4_imv_lstm_attention
   - IMV-lstm attention 실행 및 matric 평가

* 9_code_data_visualization
   - 데이터 품질 확인 / 연령, 성별에 따른 분포 확인


### Installing

Install project-related requirements in Python
(If necessary, create a virtual environment)

pip install -r requirements.txt

and

graphviz install (https://graphviz.org/download/)
- Check installation
  : cmd or terminal > "dot -V"

and 

pip install psycopg2-binary

## Getting Started

edit config.json file

* 'working_date' : Date to run the program.
* 'dbms' : mssql or postgresql
* 'mssql' or 'postgresql' : server / user /password / port .. 
* 'meas' : meas_concept_id (used by the institution)
* 'translatequerysql' : cdm_database_schema / target_database_schema / target_database_schema

### Executing program

run python script
   1-1) Execute ipynb files in the order of folder number

   or

   2-1) Execute full script
   > python main.py 
      Step by step run with (y)/(n).
   
   2-2) Run individual scripts
   > cd 0_cohort_json
   > python 0_create_cohort_person_in_db.py

### Result export
   - data (모델 생성을 위한 데이터 저장)
   - result (결과 추출을 위한 데이터 저장)
   > Compress and export the result dir

## Help

-

## Authors

suncheol heo
Researcher, DHLab, Department of Biomedical Systems Informatics,
Yonsei University College of Medicine, Seoul
mobile: (+82) 10 2789 8800
hepsie50@gmail.com

## Version History

-

## License

-

## Acknowledgments

-