# Gym Pose Aid

Projeto baseado no artigo 'Using pose estimation algorithms to build a simple gym training aid app'.
Inclui o código original (`original_ml_playground_main/`) para referência e uma nova estrutura organizada para rodar no Google Colab ou localmente.

## Estrutura
- `original_ml_playground_main/` → código original extraído do ZIP enviado.
- `notebooks/gym_pose_colab.ipynb` → notebook adaptado para Colab.
- `src/` → funções de inferência, detecção de gestos, utilidades DTW e IO.
- `scripts/` → scripts CLI para processar vídeo ou webcam.
- `out/` → saídas geradas (MP4, GIF, CSV).

## Colab
1. Abra `notebooks/gym_pose_colab.ipynb` no Google Colab.
2. Execute a primeira célula para instalar dependências.
3. Faça upload de um vídeo ou deixe o notebook baixar um exemplo.
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
