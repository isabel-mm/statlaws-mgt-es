# Cambios

## 1.0.0

Versión que acompaña a la publicación del artículo en *Linguamática*.

### Organización

- **Scripts numerados** en orden de ejecución (`01_descriptivas.py` … `12_figuras_leyes.py`), con nombres que indican qué hace cada uno.
- **Solo lo que aparece en el artículo.** Se retiran las salidas exploratorias que no forman parte de él (matrices de confusión, escala de certeza, análisis por género, importancia de rasgos y radar conjunto de modelos).
- **Metadatos de cita** (`CITATION.cff`, `.zenodo.json`) y `CHANGELOG.md`.

### Correcciones

- **Curvas ROC.** La figura se calcula con los 15 rasgos del artículo, igual que el resto de análisis; antes incluía tres rasgos más (C de Herdan, K de Yule y entropía de Shannon). Las curvas se distinguen también por el estilo de línea, no solo por el color.
- **Corrección de Benjamini-Hochberg.** Los p ajustados se calculan en orden de p creciente; antes podían quedar por debajo del p sin ajustar. La comparación global incluye las 15 variables.

## 0.1.0

Primera versión de los materiales de replicación, enviada con el manuscrito.
