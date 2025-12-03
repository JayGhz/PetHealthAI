from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import pickle
import os

app = Flask(__name__)
CORS(app)

# Obtener directorio base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Rutas de los modelos
MODEL_IMAGE_PATH = os.path.join(PROJECT_DIR, 'model', 'model_images.keras')
MODEL_TABULAR_PATH = os.path.join(PROJECT_DIR, 'model', 'model_tabular.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, 'label_encoders.pkl')
DISEASE_ENCODER_PATH = os.path.join(BASE_DIR, 'disease_encoder.pkl')

# Cargar modelos
print("Cargando modelos...")
model_image = load_model(MODEL_IMAGE_PATH)
model_tabular = load_model(MODEL_TABULAR_PATH)
print("Modelos cargados exitosamente")

# Cargar escalador y encoders (estos deben ser guardados durante el entrenamiento)
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    with open(DISEASE_ENCODER_PATH, 'rb') as f:
        disease_encoder = pickle.load(f)
    print("Escalador y encoders cargados")
except FileNotFoundError:
    print("ADVERTENCIA: Archivos de escalador/encoders no encontrados. Necesitas generarlos del notebook.")
    scaler = None
    label_encoders = None
    disease_encoder = None

# Clases de imágenes (22 clases)
IMAGE_CLASSES = [
    'Dental Disease in Cat', 'Dental Disease in Dog', 'Distemper in Dog',
    'Ear Mites in Cat', 'Eye Infection in Cat', 'Eye Infection in Dog',
    'Feline Leukemia', 'Feline Panleukopenia', 'Fungal Infection in Cat',
    'Fungal Infection in Dog', 'Hot Spots in Dog', 'Kennel Cough in Dog',
    'Mange in Dog', 'Parvovirus in Dog', 'Ringworm in Cat', 'Scabies in Cat',
    'Skin Allergy in Cat', 'Skin Allergy in Dog', 'Tick Infestation in Dog',
    'Urinary Tract Infection in Cat', 'Worm Infection in Cat', 'Worm Infection in Dog'
]

# Mapeo entre enfermedades tabulares e imágenes
DISEASE_MAPPING = {
    "Distemper": ["Distemper in Dog"],
    "Parvovirus": ["Parvovirus in Dog"],
    "Leukemia": ["Feline Leukemia"],
    "Panleukopenia": ["Feline Panleukopenia"],
    "Fungal Infection": ["Fungal Infection in Cat", "Fungal Infection in Dog"],
    "Ringworm": ["Ringworm in Cat"],
    "Cough": ["Kennel Cough in Dog"],
    "Respiratory Disease": ["Kennel Cough in Dog"],
    "Bronchitis": ["Kennel Cough in Dog"],
    "Intestinal Parasites": ["Worm Infection in Cat", "Worm Infection in Dog"]
}

# Configuración de imágenes
IMG_HEIGHT, IMG_WIDTH = 224, 224

def preprocess_image(image_bytes):
    """Preprocesa la imagen para el modelo CNN"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def preprocess_tabular_data(data):
    """Preprocesa los datos tabulares para el modelo"""
    if scaler is None or label_encoders is None:
        raise ValueError("Escalador y encoders no están disponibles")
    
    # Orden de features esperado por el modelo
    features = []
    
    # Animal_Type_encoded
    animal_type = data.get('animal_type', 'Unknown')
    features.append(label_encoders['Animal_Type'].transform([animal_type])[0])
    
    # Breed_encoded
    breed = data.get('breed', 'Unknown')
    features.append(label_encoders['Breed'].transform([breed])[0])
    
    # Age
    features.append(float(data.get('age', 0)))
    
    # Gender_encoded
    gender = data.get('gender', 'Unknown')
    features.append(label_encoders['Gender'].transform([gender])[0])
    
    # Weight
    features.append(float(data.get('weight', 0)))
    
    # Symptom_1_encoded, Symptom_2_encoded, Symptom_3_encoded, Symptom_4_encoded
    for i in range(1, 5):
        symptom = data.get(f'symptom_{i}', 'None')
        features.append(label_encoders[f'Symptom_{i}'].transform([symptom])[0])
    
    # Características binarias/numéricas
    features.append(int(data.get('appetite_loss', 0)))
    features.append(int(data.get('vomiting', 0)))
    features.append(int(data.get('diarrhea', 0)))
    features.append(int(data.get('coughing', 0)))
    features.append(int(data.get('labored_breathing', 0)))
    features.append(int(data.get('lameness', 0)))
    features.append(int(data.get('skin_lesions', 0)))
    features.append(int(data.get('nasal_discharge', 0)))
    features.append(int(data.get('eye_discharge', 0)))
    features.append(float(data.get('body_temperature', 0)))
    features.append(float(data.get('heart_rate', 0)))
    features.append(int(data.get('duration_days', 0)))
    
    # Convertir a numpy array y normalizar
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    
    return features

def get_image_prediction(image_bytes):
    """Obtiene predicción del modelo de imágenes"""
    processed_image = preprocess_image(image_bytes)
    predictions = model_image.predict(processed_image, verbose=0)[0]
    
    # Crear diccionario de predicciones
    pred_dict = {IMAGE_CLASSES[i]: float(predictions[i]) for i in range(len(IMAGE_CLASSES))}
    
    # Ordenar por probabilidad
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    
    return pred_dict, sorted_preds

def get_tabular_prediction(data):
    """Obtiene predicción del modelo tabular"""
    processed_data = preprocess_tabular_data(data)
    predictions = model_tabular.predict(processed_data, verbose=0)[0]
    
    # Crear diccionario de predicciones
    tabular_classes = disease_encoder.classes_
    pred_dict = {tabular_classes[i]: float(predictions[i]) for i in range(len(tabular_classes))}
    
    # Ordenar por probabilidad
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    
    return pred_dict, sorted_preds

def combine_predictions(image_pred_dict, tabular_pred_dict):
    """Combina predicciones de ambos modelos usando el mapeo"""
    combined_predictions = {}
    
    # Obtener enfermedad tabular más probable
    top_tabular_disease = max(tabular_pred_dict.items(), key=lambda x: x[1])[0]
    
    # Verificar si hay mapeo
    if top_tabular_disease in DISEASE_MAPPING:
        mapped_image_diseases = DISEASE_MAPPING[top_tabular_disease]
        
        # Promediar probabilidades de enfermedades mapeadas
        for image_disease in mapped_image_diseases:
            if image_disease in image_pred_dict:
                tabular_prob = tabular_pred_dict[top_tabular_disease]
                image_prob = image_pred_dict[image_disease]
                combined_predictions[image_disease] = (tabular_prob + image_prob) / 2
        
        # Si hay predicciones combinadas, retornar el resultado
        if combined_predictions:
            sorted_combined = sorted(combined_predictions.items(), key=lambda x: x[1], reverse=True)
            return sorted_combined, "combined"
    
    # Si no hay mapeo, usar solo predicción de imagen
    sorted_image = sorted(image_pred_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_image, "image_only"

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud"""
    return jsonify({"status": "ok", "models_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal de predicción"""
    try:
        # Verificar qué datos se enviaron
        has_image = 'image' in request.files
        has_tabular = 'tabular_data' in request.form or request.is_json
        
        if not has_image and not has_tabular:
            return jsonify({"error": "Se requiere al menos imagen o datos tabulares"}), 400
        
        # Caso 1: Solo imagen
        if has_image and not has_tabular:
            image_file = request.files['image']
            image_bytes = image_file.read()
            
            pred_dict, sorted_preds = get_image_prediction(image_bytes)
            
            return jsonify({
                "prediction_type": "image_only",
                "top_prediction": sorted_preds[0][0],
                "confidence": sorted_preds[0][1],
                "all_predictions": dict(sorted_preds[:5]),
                "full_predictions": pred_dict
            })
        
        # Caso 2: Solo datos tabulares
        elif not has_image and has_tabular:
            if scaler is None:
                return jsonify({"error": "Modelo tabular no está completamente configurado"}), 500
            
            if request.is_json:
                tabular_data = request.json
            else:
                import json
                tabular_data = json.loads(request.form['tabular_data'])
            
            pred_dict, sorted_preds = get_tabular_prediction(tabular_data)
            
            return jsonify({
                "prediction_type": "tabular_only",
                "top_prediction": sorted_preds[0][0],
                "confidence": sorted_preds[0][1],
                "all_predictions": dict(sorted_preds[:5]),
                "full_predictions": pred_dict
            })
        
        # Caso 3: Imagen y datos tabulares
        else:
            if scaler is None:
                # Si no hay escalador, usar solo imagen
                image_file = request.files['image']
                image_bytes = image_file.read()
                pred_dict, sorted_preds = get_image_prediction(image_bytes)
                
                return jsonify({
                    "prediction_type": "image_only",
                    "warning": "Datos tabulares ignorados (escalador no disponible)",
                    "top_prediction": sorted_preds[0][0],
                    "confidence": sorted_preds[0][1],
                    "all_predictions": dict(sorted_preds[:5]),
                    "full_predictions": pred_dict
                })
            
            image_file = request.files['image']
            image_bytes = image_file.read()
            
            if request.is_json:
                tabular_data = request.json
            else:
                import json
                tabular_data = json.loads(request.form['tabular_data'])
            
            # Obtener predicciones de ambos modelos
            image_pred_dict, _ = get_image_prediction(image_bytes)
            tabular_pred_dict, _ = get_tabular_prediction(tabular_data)
            
            # Combinar predicciones
            combined_preds, prediction_type = combine_predictions(image_pred_dict, tabular_pred_dict)
            
            return jsonify({
                "prediction_type": prediction_type,
                "top_prediction": combined_preds[0][0],
                "confidence": combined_preds[0][1],
                "all_predictions": dict(combined_preds[:5]),
                "image_predictions": dict(sorted(image_pred_dict.items(), key=lambda x: x[1], reverse=True)[:3]),
                "tabular_predictions": dict(sorted(tabular_pred_dict.items(), key=lambda x: x[1], reverse=True)[:3])
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Retorna las clases disponibles"""
    tabular_classes = disease_encoder.classes_.tolist() if disease_encoder else []
    
    return jsonify({
        "image_classes": IMAGE_CLASSES,
        "tabular_classes": tabular_classes,
        "disease_mapping": DISEASE_MAPPING
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
