#!/bin/bash
sudo chown -R vscode:vscode /workspaces/darknet
git config --global --add safe.directory /workspaces/darknet
sudo ln -s /workspaces /workers

MAKEFILE="Makefile"

if [[ -f "$MAKEFILE" ]]; then
  # Se non l'abbiamo già modificato...
  if ! grep -q "LIBSO=1" "$MAKEFILE"; then
    cp "$MAKEFILE" "${MAKEFILE}.backup"
    echo "Backup of the Makefile created as ${MAKEFILE}.backup."

    sed -i 's/GPU=0/GPU=1/g' "$MAKEFILE"
    sed -i 's/CUDNN=0/CUDNN=1/g' "$MAKEFILE"
    # Suggerimento: YOLO va molto più veloce con i float16 (Half Precision) sulle GPU moderne
    sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/g' "$MAKEFILE"
    sed -i 's/OPENCV=0/OPENCV=1/g' "$MAKEFILE"
    sed -i 's/LIBSO=0/LIBSO=1/g' "$MAKEFILE" # Cruciale per avere libdarknet.so!

    echo "Updated the Makefile with the required changes."
  fi

  echo "Compiling with make..."
  if make -j$(nproc); then
    echo "Compilation successful!"
  else
    echo "Compilation failed. Check the output for errors."
    exit 1
  fi
else
  echo "Makefile non trovato. Assicurati di aver clonato AlexeyAB/darknet qui dentro!"
fi

pip install pyyaml easydict