#!/usr/bin/env bash
# Reproduce todos los análisis y figuras. Ejecutar desde la raíz del repositorio.
# Requiere data/corpus_robot-talk_2026-02-04.csv (ver README).
set -euo pipefail

python scripts/descriptivas.py          # Descriptivos del panel 90 × 7
python scripts/demo.py                  # Extracción de rasgos (lento: spaCy + bootstrap)
python scripts/analisis_estadistico.py  # Mann-Whitney / Wilcoxon + tamaño del efecto
python scripts/clasificador.py          # T1 binaria, T2 multiclase, certeza (sobrescribe estadisticas_por_texto.csv con la columna author)
python scripts/ablacion.py              # Ablación por grupos de rasgos
python scripts/fpr_ablacion.py          # Tasa de falsos positivos (humano → robótico)
python scripts/baseline_delta.py        # Línea base Delta de Burrows
python scripts/chargram_baseline.py     # Línea base n-gramas de caracteres + F1 por clase en T2
python scripts/sensitivity_ventana.py   # Sensibilidad al tamaño de ventana
python scripts/generar_figuras.py       # Figuras de resultados
python scripts/radar_individuales.py    # Radares humano vs. cada modelo
python scripts/figures_leyes.py         # Figuras ilustrativas de Zipf, Heaps y brevedad
