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

  // 🔹 Reproduce el audio desde base64
  const handlePlay = (id) => {
    const alert = alerts.find(alert => alert.ID === id);
    if (!alert || !alert.Audio) {
      console.error("⚠️ No se encontró audio para esta alerta.");
      return;
    }
  
    try {
      setPlayingRow(id);
  
      const base64 = alert.Audio.startsWith("data:")
        ? alert.Audio.split(",")[1]
        : alert.Audio;
  
      const binaryString = atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
  
      const blob = new Blob([bytes], { type: "audio/wav" });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  
      const reader = new FileReader();
      reader.onload = function () {
        audioContext.decodeAudioData(reader.result, (buffer) => {
          const source = audioContext.createBufferSource();
          source.buffer = buffer;
  
          const gainNode = audioContext.createGain();
          gainNode.gain.value = 20.0; // Aumenta volumen (1.0 = normal, 2.0 = +6dB aprox)
  
          source.connect(gainNode);
          gainNode.connect(audioContext.destination);
  
          source.start(0);
          source.onended = () => {
            setPlayingRow(null);
            audioContext.close();
          };
        }, (error) => {
          console.error("❌ Error al decodificar audio:", error);
          setPlayingRow(null);
        });
      };
  
      reader.readAsArrayBuffer(blob);
    } catch (error) {
      console.error("❌ Error al reproducir el audio:", error);
      setPlayingRow(null);
    }
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
              <td class='maquinote'>{alert.Máquina}</td>
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
