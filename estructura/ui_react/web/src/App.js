import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import AlertTable from "./components/AlertTable";
import Graphs from "./components/Graphs";
import Filters from "./components/Filters";
import { fetchAlerts } from "./services/api"; 
import 'leaflet/dist/leaflet.css';
import MapComponent from "./components/GaliciaMap";
import "./styles/main.css";

const App = () => {
    const [filters, setFilters] = useState({ 
        estado: "Activas", 
        tipo: "Todos", 
        ubicacion: "Todas", 
        mes: new Date().toLocaleString("en-US", { month: "short" }) // ✅ Guardamos el mes en formato "Jan"
    });

    const [alerts, setAlerts] = useState([]); // Datos filtrados
    const [alertsSinFiltrar, setAlertsSinFiltrar] = useState([]); // Todos los datos

    useEffect(() => {
        console.log("🎯 Filtros actuales:", filters);
        
        fetchAlerts(filters.estado).then((data) => {
            console.log("🔍 Datos obtenidos de fetchAlerts:", data);
            if (!data || data.length === 0) {
            console.warn("⚠️ La API no está devolviendo datos.");
            }
            setAlertsSinFiltrar(data);
        });
    }, [filters.estado]); // 🔥 Solo recargar cuando cambie el estado de la alerta

    console.log("📅 Fechas en alertsSinFiltrar:", alertsSinFiltrar.map(alert => alert.Fecha_hora));

    useEffect(() => {
        console.log("🔎 Aplicando filtros...", filters);
        
        const filteredData = alertsSinFiltrar.filter(alert => 
            (filters.tipo === "Todos" || alert.Tipo === filters.tipo) &&
            (filters.ubicacion === "Todas" || alert.Ubicación === filters.ubicacion) &&
            (new Date(alert.Fecha_hora).toLocaleString("en-US", { month: "short" }) === filters.mes)
        );

        console.log("✅ Datos filtrados:", filteredData);
        setAlerts(filteredData);
    }, [filters, alertsSinFiltrar]); // 🔥 Se ejecuta cuando cambia `filters` o `alertsSinFiltrar`

    // ✅ Función para actualizar los filtros globalmente
    const handleFilterChange = (newFilters) => {
        console.log("🔄 Actualizando filtros...", newFilters);
        setFilters(prevFilters => ({ ...prevFilters, ...newFilters }));
    };

    return (
        <div className="container">
            <Header />
            
            {/* 🔹 Filtros ahora están separados y afectan AlertTable y Graphs */}
            <Filters onFilterChange={handleFilterChange} />

            <AlertTable alerts={alerts} setFilters={setFilters} setAlerts={setAlerts} />
            <Graphs alertData={alerts} fullAlertData={alertsSinFiltrar} filteredMonth={filters.mes} />

            {/* <h2>Mapa de Alertas</h2> */}
            <MapComponent fullAlertData={alertsSinFiltrar} />       
        </div>
    );
};

export default App;


