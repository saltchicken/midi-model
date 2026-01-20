#!/bin/bash

# Default values
SEARCH_ROOT="lightning_logs"
FILTER_LORA=""
FILTER_STEP=""

# Argument Parsing
while [[ $# -gt 0 ]]; do
  case $1 in
  -l | --lora)
    FILTER_LORA="$2"
    shift 2
    ;;
  -s | --step)
    FILTER_STEP="$2"
    shift 2
    ;;
  -r | --root)
    SEARCH_ROOT="$2"
    shift 2
    ;;
  *)
    # Fallback: Assume positional argument is the search root (backward compatibility)
    SEARCH_ROOT="$1"
    shift
    ;;
  esac
done

# Check if directory exists
if [ ! -d "$SEARCH_ROOT" ]; then
  echo "Error: Directory '$SEARCH_ROOT' does not exist."
  echo "Usage: ./scripts/play_training_progress.sh [-r root_dir] [-l lora_name] [-s step_number]"
  exit 1
fi

echo "🔍 Scanning for training samples in: $SEARCH_ROOT"

# Construct Find Pattern based on LoRA filter
FIND_PATTERN="*/sample"
if [ -n "$FILTER_LORA" ]; then
  FIND_PATTERN="*${FILTER_LORA}*/sample"
  echo "🎯 Filtering by LoRA name containing: '$FILTER_LORA'"
fi

if [ -n "$FILTER_STEP" ]; then
  echo "🎯 Filtering for specific step: '$FILTER_STEP'"
fi

# 1. Find all 'sample' directories matching the pattern
# 2. Sort them by path name so versions play in order
find "$SEARCH_ROOT" -type d -path "$FIND_PATTERN" | sort | while read -r sample_dir; do

  # Check step filter before printing the header
  if [ -n "$FILTER_STEP" ]; then
    if [ ! -d "$sample_dir/$FILTER_STEP" ]; then
      continue
    fi
    # If filtering by step, explicitly set steps to just that one
    steps="$FILTER_STEP"
  else
    # Existing logic: List all numeric directories
    steps=$(ls "$sample_dir" | grep -E '^[0-9]+$' | sort -n)
  fi

  if [ -z "$steps" ]; then
    continue
  fi

  echo ""
  echo "========================================================"
  echo "📂 Found Sample Directory: $sample_dir"
  echo "========================================================"

  for step in $steps; do
    step_path="$sample_dir/$step"

    # Check if there are actually .mid files inside before trying to play
    if ls "$step_path"/*.mid 1>/dev/null 2>&1; then
      echo ""
      echo "--------------------------------------------------------"
      echo "🎹 Training Step: $step"
      echo "📍 Path: $step_path"
      echo "--------------------------------------------------------"

      for midi_file in "$step_path"/*.mid; do
        filename=$(basename "$midi_file")
        echo "▶️  Playing: $filename"
        echo "   (Ctrl+C to Skip, Double Ctrl+C to Quit)"

        # Run timidity on the file
        timidity "$midi_file"

        # Capture exit code to handle Ctrl+C (SIGINT = 130 or 2)
        ret=$?

        # If interrupted (Ctrl+C), treat as SKIP, but allow quitting with a second press
        if [ $ret -eq 130 ] || [ $ret -eq 2 ]; then
          echo "⏭️  Skipped. (Press Ctrl+C again within 1s to Quit)"
          # ‼️ Increased sleep to 1s as requested to fix hanging issues
          sleep 1 || exit 1
        else
          # ‼️ Add a small buffer between songs to prevent TTY issues
          sleep 0.5
        fi
      done
    fi
  done
done

echo ""
echo "✅ Finished playing samples."
