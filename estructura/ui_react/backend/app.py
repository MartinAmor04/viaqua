from flask import Flask, jsonify, request
from flask_cors import CORS
from db_queries import get_alerts, edit_alert, add_alert
import os

app = Flask(__name__)
CORS(app)  # Habilita comunicación con React

@app.route("/api/alerts", methods=["GET"])
def obtener_alertas():
    """Obtiene alertas desde la base de datos con filtros opcionales."""
    estado = request.args.get("estado", "Pendiente")
    data = get_alerts(estado)
    return jsonify(data)

@app.route("/api/edit-alert", methods=["POST"])
def editar_alerta():
    """Permite editar una alerta específica."""
    data = request.json
    success = edit_alert(data["id"], data["alert_type"], data["estado"])
    return jsonify({"success": success})

@app.route("/api/add-alert", methods=["POST"])
def agregar_alerta():
    """Inserta una nueva alerta en la base de datos."""
    data = request.json
    success = add_alert(data["machine_id"], data["audio_record"])
    return jsonify({"success": success})

@app.route("/api/read-alert-file", methods=["GET"])
def read_alert_file():
    print('aquiiiii')
    """Lee los datos numéricos de un archivo y los devuelve como lista."""
    filename = request.args.get("filename")
    print(filename)
    file_path = os.path.join('historic_sound_data', filename)
    print(file_path)
    try:
        with open(file_path, "r") as f:
            values = [float(line.strip()) for line in f if line.strip()]
        return jsonify({"success": True, "data": values})
    except Exception as e:
        print(e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
