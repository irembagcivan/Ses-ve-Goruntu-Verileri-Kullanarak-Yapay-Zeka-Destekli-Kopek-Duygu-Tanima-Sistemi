#!/bin/zsh
unsetopt BG_NICE
cd -- "${0:A:h}"
canine_python="${CANINE_PYTHON:-${PWD}/.venv/bin/python}"
if [[ ! -x "$canine_python" ]]; then
  canine_python="${PWD:h:h}/canine-multimodal-emotion/.venv/bin/python"
fi
if [[ ! -x "$canine_python" ]]; then
  canine_python=$(command -v python3)
fi
if [[ -z "$canine_python" ]]; then
  echo 'Python 3.11 kurup .venv ortamını hazırla veya CANINE_PYTHON değişkeniyle ortamı belirt.'
  exit 1
fi
exec "$canine_python" -m src.demo.launch
