@echo off
:: Local Python shim — delegates to py -3 (Python 3.13, has cv2/numpy/ultralytics)
:: This file lives at the project root so that `python ...` works in this project
:: without modifying the system PATH. The system `python` resolves to msys64
:: which does NOT have the required packages installed.
:: Usage: python tests\benchmark_ocean.py  (just works via this shim)
py -3 %*
