@echo off


setlocal EnableExtensions EnableDelayedExpansion

set filepath1=\0_cohort\0_create_cohort_person_in_db
set filepath2=\1_importsql\0_readDB
set filepath3=\2_preprocessing_xgboost\0_preprocessing
set filepath4=\3_xgboost_classification\0_xgboost
set filepath5=\4_preprocessing_lstm\0_preprocessing_lstm
set filepath6=\5_bi-lstm_attention_classification\0_lstm_attention
set filepath7=\9_code_data_visualization\0_data_visualization

if "%$ecbId%" == "" (
    echo Enter '1' to py
    echo Enter '2' to ipynb
    echo Enter anything else to abort.
    echo/
    set "UserChoice=abort"
    set /P "UserChoice=Type input: "
    if "!UserChoice!" == "1" (
        echo toto1
		set ext1=.ipynb
		set ext2=.py
    ) else if "!UserChoice!" == "2" (
        echo toto2
        set ext1=.py
		set ext2=.ipynb
    ) else (
        echo Unknown input ... Aborting script
        endlocal
        exit /B 400
    )
)

ipynb-py-convert "%cd%%filepath1%%ext1%" "%cd%%filepath1%%ext2%"
echo %filepath1%%ext1%
ipynb-py-convert "%cd%%filepath2%%ext1%" "%cd%%filepath2%%ext2%"
echo %filepath2%%ext1%
ipynb-py-convert "%cd%%filepath3%%ext1%" "%cd%%filepath3%%ext2%"
echo %filepath3%%ext1%
ipynb-py-convert "%cd%%filepath4%%ext1%" "%cd%%filepath4%%ext2%"
echo %filepath4%%ext1%
ipynb-py-convert "%cd%%filepath5%%ext1%" "%cd%%filepath5%%ext2%"
echo %filepath5%%ext1%
ipynb-py-convert "%cd%%filepath6%%ext1%" "%cd%%filepath6%%ext2%"
echo %filepath6%%ext1%
ipynb-py-convert "%cd%%filepath7%%ext1%" "%cd%%filepath7%%ext2%"
echo %filepath7%%ext1%

endlocal

pause > nul