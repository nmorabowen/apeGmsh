; apeGmshViewer's .h5 registration (included by electron-builder.yml, nsis.include).
;
; The app joins the "Open with" list of .h5 files and never becomes their
; default handler: other tools (HDFView, h5py scripts) own .h5 too
; (maintainer decision on #1310). Everything is per-user, under HKCU, so no
; admin rights. The uninstaller removes exactly what the installer wrote,
; and the .h5 key itself when nothing else is left in it.

!define AGV_PROGID "apeGmsh.h5"
!define AGV_EXT_KEY "Software\Classes\.h5"
!define AGV_PROGID_KEY "Software\Classes\${AGV_PROGID}"

; An earlier build (V2f phase 1) made apeGmsh.h5 the .h5 default; take that
; back, and leave any other program's default alone.
!macro agvDropDefault
  ReadRegStr $0 HKCU "${AGV_EXT_KEY}" ""
  StrCmp $0 "${AGV_PROGID}" 0 +2
    DeleteRegValue HKCU "${AGV_EXT_KEY}" ""
!macroend

!macro customInstall
  !insertmacro agvDropDefault
  WriteRegStr HKCU "${AGV_PROGID_KEY}" "" "apeGmsh model, geometry or results (HDF5)"
  WriteRegStr HKCU "${AGV_PROGID_KEY}\DefaultIcon" "" "$INSTDIR\${APP_EXECUTABLE_FILENAME},0"
  WriteRegStr HKCU "${AGV_PROGID_KEY}\shell\open\command" "" '"$INSTDIR\${APP_EXECUTABLE_FILENAME}" "%1"'
  WriteRegStr HKCU "${AGV_EXT_KEY}\OpenWithProgids" "${AGV_PROGID}" ""
  ; SHCNE_ASSOCCHANGED
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
!macroend

!macro customUnInstall
  !insertmacro agvDropDefault
  DeleteRegValue HKCU "${AGV_EXT_KEY}\OpenWithProgids" "${AGV_PROGID}"
  DeleteRegKey /ifempty HKCU "${AGV_EXT_KEY}\OpenWithProgids"
  DeleteRegKey /ifempty HKCU "${AGV_EXT_KEY}"
  DeleteRegKey HKCU "${AGV_PROGID_KEY}"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
!macroend
