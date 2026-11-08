Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

strScriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
strPythonExe = strScriptDir & "\.venv\Scripts\python.exe"

If Not fso.FileExists(strPythonExe) Then
    strPythonExe = "python"
End If

strCommand = """" & strPythonExe & """ -m streamlit run """ & strScriptDir & "\streamlit_app.py"" --server.port 8501 --server.headless true"
WshShell.CurrentDirectory = strScriptDir
WshShell.Run strCommand, 0, False
