"""
Entry point for Chim-eTunes application.
"""
import sys
import os

# Ensure the 'src' directory is in the Python path
# This allows internal modules to import each other seamlessly
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from app import ChimEkenApp

if __name__ == "__main__":
    app = ChimEkenApp()
    app.mainloop()
