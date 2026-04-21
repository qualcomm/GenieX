#define MyAppName "GenieX CLI"
#define MyAppPublisher "GenieX"
#define MyAppExeName "geniex.exe"
#define MyAppIconName "geniex.ico"
#define LauncherTarget "powershell.exe"
#define LauncherArgs "-NoExit -Command geniex"

[Setup]
AppId={{e9b30237-d65d-4a79-a7c0-f4e217e78f54}}
AppName={#MyAppName}
AppVersion={#Version}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\..\
OutputBaseFilename=geniex-cli-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
SetupIconFile={#MyAppIconName}
PrivilegesRequired=lowest
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppIconName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; CLI executable (path injected via iscc /DCliExe=...)
Source: "{#CliExe}"; DestDir: "{app}"; Flags: ignoreversion
; SDK runtime libraries (path injected via iscc /DSdkDir=...)
Source: "{#SdkDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Shared icon for shortcuts and uninstall entry
Source: "{#MyAppIconName}"; DestDir: "{app}"; Flags: ignoreversion

[Registry]
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "SOFTWARE\Classes\Applications\{#MyAppExeName}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "SOFTWARE\Classes\Applications\{#MyAppExeName}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppIconName}"; Flags: uninsdeletekey

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Icons]
; All three shortcuts open PowerShell and run `geniex`; the one in {app} acts as the primary launcher.
Name: "{app}\{#MyAppName}";        Filename: "{#LauncherTarget}"; Parameters: "{#LauncherArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppIconName}"
Name: "{group}\{#MyAppName}";      Filename: "{#LauncherTarget}"; Parameters: "{#LauncherArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppIconName}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{#LauncherTarget}"; Parameters: "{#LauncherArgs}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppIconName}"; Tasks: desktopicon

[UninstallRun]
Filename: "taskkill.exe"; Parameters: "/F /IM {#MyAppExeName}"; Flags: runhidden

[Code]
const
  EnvironmentKey = 'Environment';

function InitializeSetup(): Boolean;
var
  UninstallString: String;
  ResultCode: Integer;
begin
  Result := True;
  if not RegQueryStringValue(HKCU,
       'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1',
       'UninstallString', UninstallString) then
    Exit;

  if MsgBox('Existing version detected.'#13#10 +
            'Please uninstall the existing version first.'#13#10#13#10 +
            'Uninstall now?', mbConfirmation, MB_YESNO) <> IDYES then
  begin
    MsgBox('Installation aborted.', mbInformation, MB_OK);
    Result := False;
    Exit;
  end;

  if (not Exec(RemoveQuotes(UninstallString), '/SILENT', '', SW_HIDE, ewWaitUntilTerminated, ResultCode))
     or (ResultCode <> 0) then
  begin
    MsgBox(Format('Uninstall failed (ErrCode: %d).', [ResultCode]), mbError, MB_OK);
    Result := False;
  end;
end;

procedure EnvAddPath(Path: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Paths := '';
  if Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';') > 0 then
    Exit;
  Paths := Paths + ';' + Path + ';';
  RegWriteStringValue(HKCU, EnvironmentKey, 'Path', Paths);
end;

procedure EnvRemovePath(Path: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, EnvironmentKey, 'Path', Paths) then
    Exit;
  P := Pos(';' + Uppercase(Path) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then
    Exit;
  Delete(Paths, P - 1, Length(Path) + 1);
  RegWriteStringValue(HKCU, EnvironmentKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    EnvAddPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    EnvRemovePath(ExpandConstant('{app}'));
end;
