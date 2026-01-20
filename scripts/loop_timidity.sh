while true; do
  echo "Starting new playback session"
  # ‼️ Changed to pass command line arguments (like --random or --stock) to the python script
  python random_timidity.py "$@"

  # ‼️ Brief pause to allow the user to break the loop with Ctrl+C
  # and to prevent high CPU usage if the script crashes repeatedly
  echo "--- ‼️ Session finished. Restarting in 1 seconds... (Press Ctrl+C again to quit script) ---"
  sleep 1
done
