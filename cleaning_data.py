"""
Entrypoint ringkas untuk menjalankan pipeline pembersihan data 7 platform.
Dapat dijalankan langsung dengan:
    python cleaning_data.py
atau:
    python tools/cleaning_data.py
"""
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from tools.cleaning_data import cleaning_data

if __name__ == '__main__':
    cleaning_data()
