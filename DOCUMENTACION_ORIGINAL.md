# Clasificador inicial de hojas de papa

Entrenamiento local en Windows con RTX 3060 de 12 GB; destino futuro: Raspberry Pi 5 de 2 GB.
La Raspberry no necesita las dependencias de entrenamiento de este proyecto.

## Alcance

Clasifica una fotografia centrada en una hoja de papa entre:

1. `Potato___Early_blight`: tizon temprano.
2. `Potato___Late_blight`: tizon tardio.
3. `Potato___healthy`: sana.

Es un modelo inicial de investigacion. No localiza multiples hojas en una escena,
no reconoce otras enfermedades ni rechaza automaticamente suelo, manos u objetos.
Sus puntuaciones softmax no son probabilidades clinicas calibradas.

## Datos y revision

Fuente: [PlantVillage, repositorio de los autores](https://github.com/spMohanty/PlantVillage-Dataset).
Snapshot utilizado: `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.
Solo se descargaron las imagenes originales RGB de papa: 2,152 imagenes,
aproximadamente 39.4 MB, todas de 256x256 pixeles. No se descargaron versiones
segmentadas ni en escala de grises, ni imagenes de otros cultivos.

Cada fila de `data/metadata/manifest.json` describe una imagen: ruta, etiqueta,
indice de clase, hoja de origen, hashes, dimensiones, bytes y particion.
Los nombres/rutas identifican imagenes, y `group_id` identifica hojas correlacionadas.
No se dispone de fecha individual de captura; la fecha del commit no es fecha de captura.

| Clase | Imagenes | Hojas | Entrenamiento | Validacion | Prueba |
|---|---:|---:|---:|---:|---:|
| Tizon temprano | 1,000 | 250 | 696 | 152 | 152 |
| Tizon tardio | 1,000 | 250 | 696 | 152 | 152 |
| Sana | 152 | 38 | 104 | 24 | 24 |
| Total | 2,152 | 538 | 1,496 | 328 | 328 |

Se verificaron hashes Git contra el origen, lectura completa de todas las imagenes,
etiquetas y grupos de hoja completos, ausencia de duplicados exactos de pixeles
y ausencia de grupos/hashes compartidos entre particiones. La revision visual de
15 ejemplos confirmo hojas individuales y fondos controlados; no valida por si
sola la exactitud biologica de todas las etiquetas.

La particion aproxima 70/15/15 por hoja, estratificada por clase y con semilla 42.
Cuatro imagenes de una misma hoja siempre permanecen juntas. Hay solo 6 hojas sanas
independientes en prueba (24 fotos): las metricas de esa clase tienen mucha incertidumbre.
Se guardan metricas por imagen y tambien agregadas por hoja.

## Modelo y evaluacion

MobileNetV2 parte de pesos ImageNet `IMAGENET1K_V2` de torchvision.
Se entrena primero la cabeza de 3 clases durante 5 epocas; despues se ajustan
los ultimos 3 bloques y la cabeza durante 3 epocas. Batch 32; perdida ponderada
por frecuencia inversa de clase calculada exclusivamente sobre entrenamiento.
Los aumentos de imagen se aplican solo a entrenamiento.

El mejor checkpoint se elige por macro-F1 de validacion. La prueba se evalua
una vez al terminar; no debe usarse para elegir hiperparametros.
La semilla fija favorece repetibilidad; no garantiza identidad bit a bit entre versiones/GPU.

Preprocesamiento de inferencia: RGB, redimensionar lado corto a 232, recorte central
224x224, valores 0..1, normalizar con media `[0.485,0.456,0.406]` y desviacion
`[0.229,0.224,0.225]`; tensor NCHW. Guardado tambien en `config.json` de cada ejecucion.

## Comandos en PowerShell

Entorno aislado en `.venv`, Python 3.10. Para recrearlo: `python -m venv .venv`,
actualizar pip dentro del entorno y ejecutar
`.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt`.
Las versiones de PyTorch/torchvision estan fijadas al paquete CUDA 12.8 usado aqui.

```powershell
cd C:\Users\Erick\rover-vision-training
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe train.py --head-epochs 5 --fine-epochs 3
.\.venv\Scripts\python.exe predict.py --run .\runs\CARPETA_DE_EJECUCION --image C:\ruta\hoja.jpg
```

Cada entrenamiento crea una carpeta nueva en `runs/` con:

- `config.json`: parametros, versiones, GPU, etiquetas y preprocesamiento.
- `best_weights.pt`: pesos seleccionados por validacion; no es aun un ejecutable para la Pi.
- `best_validation.json` y `history.json`: seleccion e historial.
- `test_metrics.json`: exactitud, macro-F1, precision/recall/F1 por clase y matrices de confusion.
- `test_predictions.json`: resultados individuales para revisar errores.

## Siguientes comprobaciones

1. Revisar confusion entre tizon temprano y tardio y el recall de cada clase.
2. Obtener mas hojas sanas y representar etapas leves/severas con etiquetas fiables.
3. Reunir un conjunto externo de campo: otra iluminacion, fondos, plantas y fechas;
   separar por planta/sesion antes de cualquier aumento de datos.
4. Evaluar encuadres con multiples hojas y objetos ajenos para decidir si hace falta
   detector de hojas, filtro de calidad o clase de rechazo.
5. Exportar un modelo de inferencia y medir latencia, RAM y equivalencia numerica
   en la Raspberry antes de integrarlo al rover.

Una buena puntuacion en PlantVillage no demuestra rendimiento en el cultivo real.
El [articulo original](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2016.01419/full)
documenta una caida importante al cambiar las condiciones de las imagenes.

Referencias: [modelo y pesos](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html),
[instalacion de PyTorch](https://pytorch.org/get-started/locally/).
Conservar la atribucion de la fuente y consultar sus condiciones antes de redistribuir los datos.
