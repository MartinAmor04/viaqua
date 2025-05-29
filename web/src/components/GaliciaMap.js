import React from "react";
import "../styles/Map.css";

const locations = {
  "A Coruña": { x: 120, y: 50 },
  "Vigo": { x: 70, y: 200 },
  "Pontevedra": { x: 90, y: 180 },
  "Ourense": { x: 140, y: 230 },
  "Lugo": { x: 150, y: 80 },
  "Santiago de Compostela": { x: 130, y: 120 }
};

const MapComponent = ({ fullAlertData = [], selectedLocation }) => {
  // 🔹 Filtrar alertas según la ubicación seleccionada
  const filteredAlerts = selectedLocation === "Todas"
    ? fullAlertData
    : fullAlertData.filter(alert => alert.Ubicación === selectedLocation);

  return (
    <div className="map-container">
      <img src="/Galicia.png" alt="Mapa de Galicia" className="map-image" />

      {filteredAlerts.length > 0 &&
        filteredAlerts.map((alert, index) => (
          locations[alert.Ubicación] && (
            <div
              key={index}
              className="map-point"
              style={{
                left: `${(locations[alert.Ubicación].x / 500) * 100}%`,
                top: `${(locations[alert.Ubicación].y / 400) * 100}%`,
              }}
            ></div>
          )
        ))}
    </div>
  );
};

export default MapComponent;



