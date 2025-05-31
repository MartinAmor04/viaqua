import React, { useState, useEffect, useRef } from "react";
import { editAlert, getSoundFile } from "../services/api";
import "../styles/AlertTable.css";
import Chart from "chart.js/auto";

const AlertTable = ({ alerts, setAlerts }) => {
  const [editRow, setEditRow] = useState(null);
  const [playingRow, setPlayingRow] = useState(null);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const chartRef = useRef(null);
  const lineChartInstance = useRef(null); // Para limpiar el gráfico anterior

  const handleEdit = (id) => {
    setEditRow(id);
  };

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

  const handlePlay = (id, e) => {
    e.stopPropagation();
    const alert = alerts.find(alert => alert.ID === id);
    if (!alert || !alert.Audio) return console.error("⚠️ No se encontró audio para esta alerta.");

    try {
      setPlayingRow(id);
      const base64 = alert.Audio.startsWith("data:") ? alert.Audio.split(",")[1] : alert.Audio;
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);

      const blob = new Blob([bytes], { type: "audio/wav" });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();

      const reader = new FileReader();
      reader.onload = () => {
        audioContext.decodeAudioData(reader.result, (buffer) => {
          const source = audioContext.createBufferSource();
          source.buffer = buffer;
          const gainNode = audioContext.createGain();
          gainNode.gain.value = 20.0;
          source.connect(gainNode);
          gainNode.connect(audioContext.destination);
          source.start(0);
          source.onended = () => {
            setPlayingRow(null);
            audioContext.close();
          };
        }, error => {
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

  const handleRowClick = (alert) => {
    setSelectedAlert(alert);
  };

  const closeModal = () => {
    setSelectedAlert(null);
  };

  const issueTypes = ["Rodamientos", "Fallo de fase", "Sobrecalentamiento", "Fallo mecánico", "Fallo eléctrico", "Válvula dañada"];

  useEffect(() => {
    if (selectedAlert && chartRef.current) {
      const fetchData = async () => {
        try {
          const response = await getSoundFile(`rms_machine_example.txt`);
          const data = response.data;
          console.log(data)
          if (lineChartInstance.current) {
            lineChartInstance.current.destroy();
          }

          const ctx = chartRef.current.getContext("2d");
          const filteredData = data.filter((_, index) => index % 2 === 0);
          const filteredLabels = Array.from({ length: filteredData.length }, (_, i) => i);

          lineChartInstance.current = new Chart(ctx, {
            type: "line",
            data: {
              labels: filteredLabels, // No hay eje X, solo índices
              datasets: [
                {
                  label: "Valores de RMS",
                  data: filteredData,
                  borderColor: '#fe6b13',
                  fill: false,
                  pointRadius: 0,
                },
              ],
            },
            options: {
              responsive: false,
              maintainAspectRatio: false,
              scales: {
                x: {
                  ticks: {
                    maxTicksLimit: 3, // Muestra máximo 3 etiquetas en X
                  },
                  title: {
                    display: true,
                    text: 'Índice',
                  },
                },
                y: {
            
                  title: {
                    display: true,
                    text: 'Valor RMS',
                  },
                },
              },
            },
          });
        } catch (error) {
          console.error("Error al obtener los datos:", error);
        }
      };

      fetchData();
    }
  }, [selectedAlert, alerts]);

  return (
    <div className="alert-table-container">
      <h2 className="table-title">Xestión de alertas</h2>

      <table className="alert-table">
        <thead>
          <tr>
            <th>Máquina</th>
            <th>Fecha/Hora</th>
            <th>Ubicación</th>
            <th>Avería</th>
            <th>Estado</th>
            <th>Acción</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => (
            <tr key={alert.ID} >
              <td className='maquina' onClick={() => handleRowClick(alert)} style={{ cursor: "pointer" }}><span className='maquinote'>{alert.Máquina}</span><br></br>{alert.Tipo}</td>
              <td>{alert.Fecha_hora}</td>
              <td>{alert.Ubicación}</td>

              {editRow === alert.ID ? (
                <>
                  <td>
                    <select defaultValue={alert.Tipo_avería} onChange={(e) => alert.Tipo_avería = e.target.value}>
                      {issueTypes.map(type => <option key={type} value={type}>{type}</option>)}
                    </select>
                  </td>
                  <td>
                    <select defaultValue={alert.Estado} onChange={(e) => alert.Estado = e.target.value}>
                      {["Pendiente", "En revisión", "Arreglada"].map(status =>
                        <option key={status} value={status}>{status}</option>
                      )}
                    </select>
                  </td>
                  <td>
                    <button className="save-btn" onClick={() => handleSave(alert.ID, alert.Tipo_avería, alert.Estado)}>
                      v Guardar
                    </button>
                  </td>
                </>
              ) : (
                <>
                  <td>{alert.Tipo_avería}</td>
                  <td>{alert.Estado}</td>
                  <td className="action-column">
                    <button className="action-btn" onClick={(e) => handlePlay(alert.ID, e)}>
                      {playingRow === alert.ID ? "🎧 Escuchando..." : <img src={require("../styles/img/play.png")} alt="Escuchar avería" />}
                    </button>
                    <button className="action-btn" onClick={(e) => { e.stopPropagation(); handleEdit(alert.ID); }}>
                      <img src={require("../styles/img/edit.png")} alt="Editar" />
                    </button>
                  </td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>

      {/* Popup */}
      {selectedAlert && (
        <div className="modal-overlay">
          <div className="modal-content">
            <button className="modal-close" onClick={closeModal}>✖</button>
            <h3>Detalles de la alerta</h3>
            <p><strong>Máquina:</strong> {selectedAlert.Máquina}</p>
            <canvas ref={chartRef} style={{ width: "1000px", height: "400px", marginTop: "20px" }} />
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertTable;
