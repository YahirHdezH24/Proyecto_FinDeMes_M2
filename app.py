from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_clima_regresion_v2.pkl')
scaler = joblib.load('standard_scaler_v2.pkl')  # Asegúrate de que 'scaler.pkl' sea un objeto de escalado como StandardScaler o MinMaxScaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        engine_size = float(request.form['ENGINESIZE'])
        cylinders = float(request.form['CYLINDERS'])
        fuel_type = float(request.form['FUELTYPE'])  # Asegúrate de que el valor recibido sea numérico
        fuel_consumption_comb = float(request.form['FUELCONSUMPTION_COMB'])

        # Escalar los datos de entrada
        input_data = [[engine_size, cylinders, fuel_type, fuel_consumption_comb]]
        scaled_data = scaler.transform(input_data)
        
        # Crear un DataFrame con los datos escalados
        data_df = pd.DataFrame(scaled_data, columns=['ENGINESIZE', 'CYLINDERS', 'FUELTYPE', 'FUELCONSUMPTION_COMB'])
        
        # Imprimir el DataFrame para verificar los datos
        print("Datos recibidos (escalados):")
        print(data_df)
        
        # Realizar la predicción
        prediction = model.predict(data_df)[0]
        
        # Devolver la predicción como respuesta JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
