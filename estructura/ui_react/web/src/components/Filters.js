import React, { useState, useEffect } from "react";
import "../styles/Filters.css";

const Filters = ({ onFilterChange }) => {
    // 🔹 Obtener el mes actual en formato "Jan"
    const currentMonth = new Date().toLocaleString("en-US", { month: "short" });

    const [estado, setEstado] = useState("Activas");
    const [tipo, setTipo] = useState("Todos");
    const [ubicacion, setUbicacion] = useState("Todas");
    const [mes, setMes] = useState(currentMonth);

    // 🔹 Usamos useEffect para enviar los filtros actualizados al inicio
    useEffect(() => {
        onFilterChange({ estado, tipo, ubicacion, mes });
    }, [estado, tipo, ubicacion, mes]);

    // 🔹 Función genérica para manejar cambios en los filtros
    const handleFilterChange = (event, filterKey) => {
        const newValue = event.target.value;

        console.log(`🔄 Cambio en filtro: ${filterKey} → ${newValue}`);
        
        // 🔥 Actualizar estado local
        if (filterKey === "estado") setEstado(newValue);
        if (filterKey === "tipo") setTipo(newValue);
        if (filterKey === "ubicacion") setUbicacion(newValue);
        if (filterKey === "mes") setMes(newValue);

        // 🔥 Enviar actualización a `App.js`
        onFilterChange(prevState => ({ ...prevState, [filterKey]: newValue }));
    };

    return (
        <div>
            <h2 className="graph-title">Filtros</h2>
            <div className="filters">
                <div>
                    <label>Estado:</label>
                    <select value={estado} onChange={(e) => handleFilterChange(e, "estado")}>
                        <option value="Todas">Todas</option>
                        <option value="Pendiente">Pendiente</option>
                        <option value="Arreglada">Arreglada</option>
                    </select>
                </div>
                
                <div>
                    <label>Tipo:</label>
                    <select value={tipo} onChange={(e) => handleFilterChange(e, "tipo")}>
                        <option value="Todos">Todos</option>
                        <option value="Soplante">Soplante</option>
                        <option value="Compresor">Compresor</option>
                        <option value="Generador">Generador</option>
                        <option value="Motor">Motor</option>
                        <option value="Bomba">Bomba</option>
                    </select>
                </div>
                <div>
                    <label>Mes:</label>
                    <select value={mes} onChange={(e) => handleFilterChange(e, "mes")}>
                        {["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map(m =>
                            <option key={m} value={m}>{m}</option>
                        )}
                    </select>
                </div>
            </div>
        </div>
    );
};

export default Filters;

