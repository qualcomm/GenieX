#define MyAppName "GenieX CLI"
#define MyAppPublisher "GenieX"
#define MyAppExeName "geniex.exe"
#define MyAppIconName "geniex.ico"
#define MyAppGuid "e9b30237-d65d-4a79-a7c0-f4e217e78f54"
#define LauncherTarget "powershell.exe"
#define LauncherArgs "-NoExit -Command geniex"

[Setup]
AppId={{{#MyAppGuid}}
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
#include ArtifactIss
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

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
const
  EnvironmentKey = 'Environment';

function InitializeSetup(): Boolean;
var
  UninstallKey, UninstallString, Dummy: String;
  ResultCode, Waited: Integer;
begin
  Result := True;
  UninstallKey := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{{#MyAppGuid}}_is1';
  if not RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', UninstallString) then
    Exit;

  if MsgBox('Existing version detected.'#13#10 +
            'Please uninstall the existing version first.'#13#10#13#10 +
            'Uninstall now?', mbConfirmation, MB_YESNO) <> IDYES then
  begin
    MsgBox('Installation aborted.', mbInformation, MB_OK);
    Result := False;
    Exit;
  end;

  if (not Exec(RemoveQuotes(UninstallString), '/SILENT', '', SW_SHOW, ewWaitUntilTerminated, ResultCode))
     or (ResultCode <> 0) then
  begin
    MsgBox(Format('Uninstall failed (ErrCode: %d).', [ResultCode]), mbError, MB_OK);
    Result := False;
    Exit;
  end;

  Waited := 0;
  while RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', Dummy) do
  begin
    if Waited >= 30000 then
    begin
      MsgBox('Timed out waiting for the previous version to finish uninstalling.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    Sleep(500);
    Waited := Waited + 500;
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
