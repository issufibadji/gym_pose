# Gym Pose Aid

Projeto mínimo para rodar o MoveNet e detectar gestos simples em vídeos ou webcam.

## Colab
1. Faça upload do diretório `gym_pose_aid` ou abra o notebook `notebooks/gym_pose_colab.ipynb`.
2. Execute a primeira célula para instalar dependências.
3. Carregue um vídeo ou deixe o notebook baixar um exemplo.
4. Rode o processamento para gerar `out/pose_overlay.mp4` e, se ativado, `out/events.csv`.

## Local (Windows/Linux)
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_video.py --video path/to/video.mp4 --model lightning --out out/ --events
```

Notas:
- FFmpeg é requerido para `imageio-ffmpeg`.
- GPU é opcional. Defina `TFHUB_CACHE_DIR` para cachear modelos.
- Ajuste thresholds de gestos em `src/gestures.py`.
