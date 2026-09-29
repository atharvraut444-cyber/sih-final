@echo off
:: Convenience wrapper — runs the ocean benchmark with py -3 (correct interpreter)
:: Usage: run_benchmark.bat [--n 25] [--seed 1234] [--out results.json] [--force]
py -3 tests\benchmark_ocean.py %*
