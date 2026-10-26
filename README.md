# Leyes estadísticas del lenguaje y detección de textos robóticos

Código y resultados que acompañan al artículo:

> Moyano-Moreno, I. **Las leyes estadísticas del lenguaje como marcador de estilo para la detección de textos robóticos en español** / *Statistical Laws of Language as a Stylometric Fingerprint for Machine-Generated Text Detection in Spanish*. Enviado a *Linguamática*.

El trabajo evalúa 15 rasgos estilométricos derivados de las leyes de Zipf, Heaps y el principio de brevedad para distinguir artículos científicos en español escritos por humanos de los generados por seis LLMs (Bard, Claude 3.5 Sonnet, DeepSeek-V3, GPT-3.5-turbo, GPT-4 y Mixtral-8x7B), sobre un panel pareado de 90 artículos × 7 autores (630 textos) del corpus ROBOT-TALK.

## Estructura

```
data/       Aquí debe colocarse el corpus (no incluido)
scripts/    Código Python de todos los análisis
results/    Tablas CSV con los resultados del artículo
figures/    Figuras del artículo (PDF y PNG), regenerables con los scripts
run_all.sh  Ejecuta todo el flujo en el orden correcto
```

## Corpus

Los análisis usan el corpus **ROBOT-TALK** (Alonso et al., 2024). El fichero del corpus **no se distribuye** en este repositorio. Para reproducir el flujo completo, solicite acceso a los autores del corpus y coloque el fichero en:

```
data/corpus_robot-talk_2026-02-04.csv
```

Sin el corpus pueden reproducirse igualmente todos los análisis posteriores a la extracción de rasgos, ya que `results/estadisticas_textos.csv` contiene los rasgos calculados (sin texto) para las 100 ventanas de cada uno de los 630 textos.

## Instalación

Python 3.11:

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Reproducción

Todos los scripts se ejecutan **desde la raíz del repositorio**:

```bash
./run_all.sh
```

| Paso | Script | Necesita corpus | Salida |
|---|---|---|---|
| 1 | `descriptivas.py` | sí | `results/descriptivas_articulos_90x7.csv` |
| 2 | `demo.py` | sí | `results/estadisticas_textos.csv` (rasgos por ventana) |
| 3 | `analisis_estadistico.py` | no | `results/analisis_*.csv` |
| 4 | `clasificador.py` | no | T1/T2, `clasificacion_multiclase_autor.csv`, `certainty_*`, `confusion_*.png`, `estadisticas_por_texto.csv` |
| 5 | `ablacion.py` | no | `results/ablacion.csv` |
| 6 | `fpr_ablacion.py` | no | `results/fpr_ablacion.csv` |
| 7 | `baseline_delta.py` | sí | `results/baseline_delta.csv` |
| 8 | `chargram_baseline.py` | sí | `results/t2_perclass_f1.csv` |
| 9 | `sensitivity_ventana.py` | sí | `results/sensitivity_ventana.csv` |
| 10 | `generar_figuras.py` | no | figuras de resultados en `figures/` |
| 11 | `radar_individuales.py` | no | `figures/radar_individuales.*` |
| 12 | `figures_leyes.py` | sí | `figures/ley_{zipf,heaps,brevedad}.*` |

`functions.py` contiene las funciones de cálculo de los rasgos (pendiente de Zipf, exponente de Heaps, pendiente de brevedad, diversidad léxica, entropía, etc.).

Notas:
- El orden importa: `clasificador.py` debe ejecutarse después de `analisis_estadistico.py`, porque ambos escriben `results/estadisticas_por_texto.csv` y `chargram_baseline.py` necesita la versión de `clasificador.py` (con la columna `author`).
- La semilla aleatoria es fija (`seed = 42`) en el muestreo de ventanas y en las particiones de validación cruzada.
- La validación cruzada usa `GroupKFold` (k = 5) agrupando por artículo, de modo que las versiones de un mismo artículo nunca están a la vez en entrenamiento y prueba.

## Licencia

Código y resultados bajo licencia MIT (ver `LICENSE`). El corpus ROBOT-TALK no forma parte de este repositorio y se rige por sus propias condiciones.

## Contacto

Isabel Moyano-Moreno — Instituto Universitario de Investigación en Lingüística Aplicada, Universidad de Cádiz — isabel.moyano@uca.es

## Agradecimientos

Proyecto «ROBOT-TALK: Reconocimiento del origen robótico de textos. Automatización de tareas y conocimiento lingüístico» (PID2022-140897OB-I00), financiado por MCIN/AEI/10.13039/501100011033 y por la Unión Europea (FEDER). Ayuda predoctoral FPU21/03261 del Ministerio de Universidades.
