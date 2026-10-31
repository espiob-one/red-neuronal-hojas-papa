# Red neuronal para hojas de papa

Código del proyecto de visión del rover: clasificación de hojas sanas, tizón temprano y tizón tardío. Incluye los 16 archivos del paquete de capturas y sus módulos locales de apoyo.

## Archivos principales

- `prepare_data.py`: preparación y partición del dataset.
- `train.py`: entrenamiento original de MobileNetV2.
- `train_robust_classifier.py` y `train_motion_classifier.py`: experimentos de robustez y movimiento.
- `train_leaf_verifier.py`: filtro hoja/no hoja.
- `train_potato_detector.py`: entrenamiento del detector.
- `train_print_classifier.py` y `train_print_head_classifier.py`: adaptación experimental para impresos.
- `evaluate_*.py`: evaluación.
- `export_*.py`: exportación a ONNX.

## Uso y alcance

Consulta `DOCUMENTACION_ORIGINAL.md` para el procedimiento del entrenamiento inicial y la procedencia de PlantVillage. Las dependencias originales se conservan en `requirements-training.txt` y `requirements-live.txt`; los experimentos adicionales pueden requerir otras bibliotecas.

Este repositorio conserva código fuente. Los datasets, capturas privadas, pesos y resultados históricos no están incluidos. Los experimentos avanzados requieren los manifiestos y checkpoints referenciados en sus rutas; no funcionan desde cero únicamente al clonar este repositorio. La adaptación para impresos conocidos no demuestra precisión independiente en campo.

Se verificó la sintaxis de los archivos al preparar esta copia; no se repitió el entrenamiento.
