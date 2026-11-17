#!/usr/bin/env bash
# Ejecuta los análisis y genera las figuras, en orden.
# Ejecutar desde la raíz del repositorio. Los pasos marcados con [corpus]
# necesitan data/corpus_robot-talk_2026-02-04.csv (ver README).
set -euo pipefail

python scripts/01_descriptivas.py          # [corpus] Descriptivos del panel 90 × 7
python scripts/02_extraccion_rasgos.py     # [corpus] Extracción de rasgos (lento: spaCy + bootstrap)
python scripts/03_analisis_estadistico.py  # Mann-Whitney / Wilcoxon + tamaño del efecto
python scripts/04_clasificacion.py         # T2: identificación multiclase de autor
python scripts/05_ablacion.py              # Ablación por grupos de rasgos
python scripts/06_falsos_positivos.py      # Tasa de falsos positivos (humano → robótico)
python scripts/07_baseline_delta.py        # [corpus] Línea base Delta de Burrows
python scripts/08_baseline_chargrams.py    # [corpus] Línea base n-gramas de caracteres + F1 por clase (T2)
python scripts/09_sensibilidad_ventana.py  # [corpus] Sensibilidad al tamaño de ventana
python scripts/10_figuras_resultados.py    # Figuras de resultados
python scripts/11_figuras_radar.py         # Radares humano vs. cada modelo
python scripts/12_figuras_leyes.py         # [corpus] Figuras ilustrativas de Zipf, Heaps y brevedad
