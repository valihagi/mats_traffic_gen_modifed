@ECHO off
SET cameo_path=C:\CAMEO\Cameo4_R2\Hotfix Development\Development\Libraries
REM set license file/server path
SET USER_LICENSE_FILE=C:\CAMEO\Cameo4_R2\Hotfix Development\Development\Libraries;27000@ATGRZSW2891
SET PORT=9842

REM check if the port is accessible 
FOR /f "usebackq delims=" %%i in (`
	powershell -noprofile -c "(New-Object System.Net.Sockets.TcpClient).ConnectAsync('localhost',%PORT%).Wait(1500)"
`) DO SET is_open="%%i"

IF %is_open% == "True" (
	ECHO "already running"
	pause
) ELSE (
	ECHO "not running. start it"
    "%cameo_path%\Python\python.exe" "%cameo_path%\Services\faas\sg.py" -c "%~dp0active_doe.standalone.config.yaml" --log-dir "%ProgramData%\AVL\Cameo\log"
)
