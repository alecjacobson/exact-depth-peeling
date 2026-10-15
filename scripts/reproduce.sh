#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
python scripts/generate_fixtures.py
ctest --test-dir build --output-on-failure
# Separate suite budgets keep the costly cover sweep from starving other cases.
python scripts/benchmark.py --suite all --output results/benchmark --timeout 5 --budget 600
python scripts/benchmark.py --suite stacked --output results/stacks --timeout 5 --budget 180
python scripts/benchmark.py --suite other --output results/other --timeout 5 --budget 180
python scripts/benchmark.py --suite broader --output results/broader --timeout 5 --budget 180
python scripts/benchmark.py --suite other --only-workloads connected-mesh \
  --methods visible-peel envelope-peel --orders near-first --layers 1 all \
  --output results/connected --timeout 30 --budget 180
python scripts/benchmark.py --suite broader --only-workloads nested-mesh \
  --methods visible-peel envelope-peel --orders near-first --layers 1 all \
  --output results/nested --timeout 30 --budget 180
python scripts/benchmark.py --suite foreground-cover --sizes 64 --layers 1 \
  --methods visible-peel envelope-peel --output results/large-first --timeout 5 --budget 90
python scripts/benchmark.py --suite practical --methods visible-peel envelope-peel \
  --orders near-first --layers 1 2 4 --output results/practical --timeout 30 --budget 300
python scripts/plot_results.py results/benchmark --include results/large-first
python scripts/plot_results.py results/stacks
python scripts/plot_results.py results/practical
python scripts/export_examples.py
python scripts/summarize_results.py
