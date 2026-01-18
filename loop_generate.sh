while true; do
  echo "Starting new generation session"
  python generate.py \
    --model models/model.safetensors \
    --bpm random \
    --time_sig random \
    --num_events 4096 \
    --lora_strength 1.0 \
    --version random \
    --lora random \
    --best_lora

  # ‼️ Brief pause to allow the user to break the loop with Ctrl+C
  # and to prevent high CPU usage if the generation finishes instantly
  echo "--- ‼️ Generation finished. Restarting in 2 seconds... (Press Ctrl+C to stop) ---"
  sleep 2
done
