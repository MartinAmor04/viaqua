from flask import Flask, jsonify, request
from flask_cors import CORS
from backend.db_queries import get_alerts, edit_alert, add_alert

app = Flask(__name__)
CORS(app)  # Habilita comunicación con React

@app.route("/api/alerts", methods=["GET"])
def obtener_alertas():
    """Obtiene alertas desde la base de datos con filtros opcionales."""
    estado = request.args.get("estado", "Activas")
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

if __name__ == "__main__":
    app.run(debug=True)
