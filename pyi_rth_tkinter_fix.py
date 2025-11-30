import os
import sys

# Fix Tcl/Tk paths for PyInstaller
if hasattr(sys, '_MEIPASS'):
    # Running as PyInstaller bundle
    os.environ['TCL_LIBRARY'] = os.path.join(sys._MEIPASS, '_tcl_data', 'tcl8.6')
    os.environ['TK_LIBRARY'] = os.path.join(sys._MEIPASS, '_tk_data', 'tk8.6')