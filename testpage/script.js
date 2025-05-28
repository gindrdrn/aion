// script.js

// Referencias a elementos del DOM
const imagenCalleInput = document.getElementById("imagen-calle"); // Nuevo ID para el input de archivo
const imagePreview = document.getElementById("image-preview");   // Nuevo ID para la imagen de previsualización
const estadoCalleParrafo = document.getElementById("estado-calle"); // Nuevo ID para el párrafo de salida

// Tus 3 categorías de clasificación, en el orden que tu modelo las predice
const CLASS_LABELS = ["Buen Estado", "Deterioro Moderado", "Deterioro Grave"];

// Desplegar imagen subida por el usuario
imagenCalleInput.addEventListener('change', (event) => {
    if (event.target.files && event.target.files[0]) {
        imagePreview.src = URL.createObjectURL(event.target.files[0]);
        imagePreview.style.display = 'block'; // Muestra la imagen
        console.log("Imagen seleccionada:", imagePreview.src);
        estadoCalleParrafo.textContent = ''; // Limpia el resultado anterior
    } else {
        imagePreview.src = '#';
        imagePreview.style.display = 'none'; // Oculta la imagen si no hay archivo
    }
});

// Función para realizar la predicción del estado de la calle
async function predict_street_state() {
    // Verificar si se ha seleccionado una imagen
    if (!imagenCalleInput.files || imagenCalleInput.files.length === 0) {
        estadoCalleParrafo.textContent = "Por favor, selecciona una imagen para predecir.";
        return;
    }

    estadoCalleParrafo.textContent = "Prediciendo..."; // Mensaje de carga

    try {
        // Preprocesamiento de la imagen para InceptionV3 (224x224 y normalizado a [-1, 1])
        let imageproc = tf.tidy(() => {
            return tf.browser.fromPixels(imagePreview)
                .resizeNearestNeighbor([224, 224]) // Redimensiona la imagen a 224x224
                .toFloat()                           // Convierte a flotante
                .div(127.5)                          // Normaliza a [-1, 1] (dividimos por 127.5 y restamos 1)
                .sub(1.0)
                .expandDims(0);                      // Agrega la dimensión del batch
        });
        console.log("Finalización del preprocesamiento de la imagen para InceptionV3");

        // Carga el modelo (se carga cada vez que se llama a la función, como en tu ejemplo original)
        // Asegúrate de que la ruta a tu modelo sea correcta.
        // Si estás usando un modelo pre-entrenado de InceptionV3 de TensorFlow.js,
        // la carga podría ser diferente (e.g., `tf.sequential().fromLayers(...)` o usando un modelo predefinido).
        // Si tu modelo es una exportación de un InceptionV3 fine-tuneado, esta línea es correcta.
        console.time(`Tiempo de carga y predicción`);
        const model = await tf.loadLayersModel('./tensorflowjs-model/model.json'); // Asegúrate de que esta ruta sea correcta
        const pred = model.predict(imageproc);
        await pred.data(); // Espera a que la predicción se complete
        console.timeEnd(`Tiempo de carga y predicción`);
        pred.print();
        console.log("Finalización de predicción");

        // Obtener los resultados y mostrar la clase con mayor probabilidad
        const data = await pred.data();
        console.log("Resultados de la predicción:", data);

        let max_val = -1;
        let max_val_index = -1;

        for (let i = 0; i < data.length; i++) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_val_index = i;
            }
        }

        const probability = max_val;
        const confidenceThreshold = 0.4; // Puedes ajustar este umbral

        if (probability > confidenceThreshold) {
            estadoCalleParrafo.innerHTML = `<p><strong>Estado detectado:</strong> ${CLASS_LABELS[max_val_index]} (${(probability * 100).toFixed(2)}% probabilidad)</p>`;
        } else {
            estadoCalleParrafo.innerHTML = `<p>No se pudo determinar el estado con suficiente confianza (máxima probabilidad: ${(probability * 100).toFixed(2)}%).</p>`;
        }

        // Liberar memoria de tensores
        tf.dispose([imageproc, pred]);

    } catch (error) {
        console.error('Error durante la predicción:', error);
        estadoCalleParrafo.textContent = 'Error al realizar la predicción. Revisa la consola para más detalles.';
    }
}
