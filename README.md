# CNN-TC

# Proyecto de Procesamiento y Clasificación de Imágenes TC utilizando Pytorch

Este repositorio contiene una colección de Jupyter Notebooks diseñados para el procesamiento y clasificación de imágenes de Tomografía Computarizada (TC) del pulmón, utilizando Python y técnicas de aprendizaje profundo. Los notebooks están optimizados para ser ejecutados en Google Colab, aprovechando el uso de GPUs. Por otro lado, también se sugiere el uso de Kaggle.

## Documentación Adicional

Para más información sobre la investigación y comparación detallada de resultados obtenidos utilizando estos algoritmos, consulta el documento adjunto: [CORE2023_DeepLearning](CORE2023_CIC-IPN_DeepLearning.pdf).

## Contenido

1. **Segmentación de Pulmón** (`segmentación_pulmón_OpenCV_TC.ipynb`)
   - **Descripción**: Este notebook realiza la segmentación del pulmón en imágenes TC utilizando OpenCV. Procesa una carpeta de imágenes y genera una nueva con las imágenes segmentadas.
   - **Tecnologías Utilizadas**: OpenCV.

2. **Aumento de Brillo** (`aumento_Brillo_TC_OpenCV.ipynb`)
   - **Descripción**: Este notebook incrementa el brillo de imágenes TC utilizando la función `convertScaler` de OpenCV.
   - **Tecnologías Utilizadas**: OpenCV.

3. **Clasificación con InceptionV3** (`InceptionV3_Pytorch.ipynb`)
   - **Descripción**: Implementa la arquitectura InceptionV3 para la clasificación de imágenes TC. El modelo se entrena, guarda los pesos al obtener el error de validación más bajo, y luego se evalúa en el conjunto de test. Se generan métricas como F1-Score, Accuracy, Recall, Precision y una matriz de confusión estilizada como mapa de calor. Se añade 3 capas fully conected a InceptionV3 para mejorar los resultados de clasificación.
   - **Tecnologías Utilizadas**: PyTorch.

4. **GridSearch para InceptionV3** (`GridSearch_InceptionV3_Pytorch.ipynb`)
   - **Descripción**: Realiza una búsqueda de hiperparámetros óptimos para InceptionV3, iterando entre diferentes valores de learning rate y optimizadores.
   - **Tecnologías Utilizadas**: PyTorch.

5. **Clasificación con ResNet-50** (`ResNet50_Pytorch.ipynb`)
   - **Descripción**: Implementa ResNet-50 para la clasificación de imágenes TC, siguiendo un proceso similar al de InceptionV3. Se mantiene la arquitectura original.
   - **Tecnologías**: PyTorch.

6. **GridSearch para ResNet-50** (`GridSearch_ResNet50_Pytorch.ipynb`)
   - **Descripción**: Realiza una búsqueda de hiperparámetros para la arquitectura ResNet-50.
   - **Tecnologías**: PyTorch.

7. **Clasificación con VGG16** (`VGG16_Pytorch.ipynb`)
   - **Descripción**: Usa VGG16 para clasificar imágenes TC, incluyendo entrenamiento, evaluación y métricas. Se añade Batch Normalization a VGG16 para mejorar y reducir los tiempos de entrenamiento.
   - **Tecnologías**: PyTorch.

8. **GridSearch para VGG16** (`GridSearch_VGG16_Pytorch.ipynb`)
   - **Descripción**: Busca los mejores hiperparámetros para VGG16.
   - **Tecnologías**: PyTorch.

## Sobre el Dataset

Este proyecto utiliza un dataset privado para el procesamiento y análisis de imágenes. Por razones de privacidad y protección de la integridad de los pacientes, el dataset no se incluye en este repositorio. El dataset contiene imágenes de Tomografía Computarizada (TC) y ha sido utilizado bajo estrictas medidas de confidencialidad para asegurar la privacidad y seguridad de los datos de los pacientes.

### Nota Sobre el Uso del Dataset

- **Confidencialidad**: El dataset es confidencial y no está disponible públicamente.
- **Protección de Datos**: Se han tomado todas las medidas necesarias para proteger la identidad y la integridad de los datos de los pacientes.
- **Uso en el Proyecto**: Las instrucciones y código proporcionados en este repositorio están diseñados para ser utilizados con un conjunto de datos similar. Si deseas reproducir o extender este trabajo, deberás disponer de un dataset propio o utilizar un dataset público adecuado, respetando siempre las normativas y leyes de privacidad de datos.

## Instalación y Uso

Para utilizar estos notebooks:

1. Clona el repositorio en tu entorno local o abre los notebooks directamente en Google Colab.
2. Asegúrate de tener instaladas las dependencias necesarias, como OpenCV y PyTorch.
3. Sigue las instrucciones detalladas en cada notebook para procesar y analizar tus imágenes TC.

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si deseas contribuir, por favor lee las directrices de contribución y envía tus pull requests.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Contacto

Si tienes preguntas, comentarios o deseas más información sobre el proyecto, no dudes en contactar a Francisco González:

- Email: [faragong1300@alumno.ipn.mx](mailto:faragong1300@alumno.ipn.mx)
