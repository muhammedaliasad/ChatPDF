@echo off

:: Set the paths for python.exe and pip.exe
set PYTHON_PATH=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe
set PIP_PATH=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\Scripts\pip.exe

:: Navigate to the specified directory
cd /d D:\AIMSGPTGuide\ChatPDF

:: Install the packages from requirements.txt
echo Installing Python packages from requirements.txt...
%PIP_PATH% install -r requirements.txt
%PIP_PATH% install --upgrade streamlit-extras 

:: Run the Streamlit app
echo Running the Streamlit app...

%PYTHON_PATH% -m streamlit run main.py

:: Pause the command prompt so it doesn't close immediately
pause
