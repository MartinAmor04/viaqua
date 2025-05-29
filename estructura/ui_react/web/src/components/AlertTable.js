import React, { useEffect, useState } from "react";
import { editAlert } from "../services/api";
import "../styles/AlertTable.css";

const AlertTable = ({ alerts, setAlerts }) => {
  const [editRow, setEditRow] = useState(null);
  const [playingRow, setPlayingRow] = useState(null);

  // 🔹 Función para activar el modo edición
  const handleEdit = (id) => {
    setEditRow(id);
  };

  // 🔹 Función para guardar los cambios
  const handleSave = (id, newType, newStatus) => {
    editAlert(id, newType, newStatus).then((response) => {
      if (response.success) {
        setAlerts(alerts.map(alert => alert.ID === id ? { ...alert, Tipo_avería: newType, Estado: newStatus } : alert));
        setEditRow(null); 
      } else {
        console.error("❌ Error al guardar los cambios.");
      }
    });
  };

  // 🔹 Simulación de reproducción de audio
  const handlePlay = (id) => {
    setPlayingRow(id);
    setTimeout(() => setPlayingRow(null), 5000); 
  };

  const issueTypes = ["Fallo eléctrico", "Sobrecalentamiento", "Pérdida de potencia", "Fallo mecánico", "Fallo en sensor"];

  return (
    <div className="alert-table-container">
      <h2 className="table-title">Xestión de alertas</h2>

      <table className="alert-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Máquina</th>
            <th>Tipo</th>
            <th>Fecha/Hora</th>
            <th>Ubicación</th>
            <th>Avería</th>
            <th>Estado</th>
            <th>Acción</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => (
            <tr key={alert.ID}>
              <td>{alert.ID}</td>
              <td>{alert.Máquina}</td>
              <td>{alert.Tipo}</td>
              <td>{alert.Fecha_hora}</td>
              <td>{alert.Ubicación}</td>

              {editRow === alert.ID ? (
                <>
                  {/* 🔹 Desplegable para modificar el Tipo de Avería */}
                  <td>
                    <select defaultValue={alert.Tipo_avería} onChange={(e) => alert.Tipo_avería = e.target.value}>
                      {issueTypes.map(type => <option key={type} value={type}>{type}</option>)}
                    </select>
                  </td>

                  {/* 🔹 Desplegable para modificar el Estado */}
                  <td>
                    <select defaultValue={alert.Estado} onChange={(e) => alert.Estado = e.target.value}>
                      {["Pendiente", "En revisión", "Arreglada"].map(status =>
                        <option key={status} value={status}>{status}</option>
                      )}
                    </select>
                  </td>

                  {/* 🔹 Botón para guardar cambios */}
                  <td>
                    <button className="save-btn" onClick={() => handleSave(alert.ID, alert.Tipo_avería, alert.Estado)}>
                      ✅ Guardar
                    </button>
                  </td>
                </>
              ) : (
                <>
                  <td>{alert.Tipo_avería}</td>
                  <td>{alert.Estado}</td>
                  <td className="action-column">
                    {/* 🔹 Botón para reproducir el audio */}
                    <button className="action-btn" onClick={() => handlePlay(alert.ID)}>
                      {playingRow === alert.ID ? "🎧 Escuchando..." : <img src={require("../styles/img/play.png")} alt="Escuchar avería" />}
                    </button>

                    {/* 🔹 Botón para activar la edición */}
                    <button className="action-btn" onClick={() => handleEdit(alert.ID)}>
                      <img src={require("../styles/img/edit.png")} alt="Editar" />
                    </button>
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AlertTable;


