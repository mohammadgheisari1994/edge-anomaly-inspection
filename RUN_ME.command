#!/bin/bash
# Double-click in Finder to run the whole demo in Terminal.
cd "$(dirname "$0")"
if ./run_all.sh; then
  echo
  echo "SUCCESS. Results are in results/pcb1/. You can close this window."
else
  echo
  echo "FAILED. See run.log for the error."
fi
read -r -p "Press Enter to close..."
