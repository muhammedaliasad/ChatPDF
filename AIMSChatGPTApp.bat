@echo off

:: Set the paths for python.exe

set PYTHON_PATH=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe

:: Run the Streamlit app

echo Running the Streamlit app...
%PYTHON_PATH% -m streamlit run main.py

:: Pause the command prompt so it doesn't close immediately

pause
