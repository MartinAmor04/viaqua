import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { CategoryScale, ArcElement, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend } from "chart.js";
import "../styles/Graphs.css";

// Registrar elementos necesarios para Chart.js
Chart.register(CategoryScale, ArcElement, LinearScale, BarElement, LineElement, PointElement, Title, Tooltip, Legend);

function Graphs({ alertData, fullAlertData, filteredMonth }) {
  const pieChartRef = useRef(null);
  const barChartRef = useRef(null);
  const lineChartRef = useRef(null);

  useEffect(() => {
    console.log("🔄 Actualizando gráficos con filtros:");
    console.log("📊 Datos filtrados - Graphs:", alertData);

    if (!alertData.length || !fullAlertData.length) {
      console.warn("⚠️ No hay datos suficientes para mostrar gráficos.");
      return;
    }

    // ✅ Obtener todos los meses del año
    const monthsOfYear = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

    // ✅ Inicializar `alertsByMonth`
    const alertsByMonth = monthsOfYear.reduce((acc, month) => ({ ...acc, [month]: 0 }), {});

    // ✅ Procesar datos para el gráfico de barras
    fullAlertData.forEach(alert => {
        if (!alert.Fecha_hora) return;

        const alertDate = new Date(alert.Fecha_hora);
        if (isNaN(alertDate.getTime())) {
            console.warn("🚨 Fecha inválida encontrada:", alert.Fecha_hora);
            return;
        }

        const month = alertDate.toLocaleString("en-US", { month: "short" });

        if (alertsByMonth.hasOwnProperty(month)) {
            alertsByMonth[month]++;
        }
    });

    const labelsBar = monthsOfYear;
    const valuesBar = monthsOfYear.map(month => alertsByMonth[month]);

    // ✅ Usamos `filteredMonth` para destacar el mes filtrado en naranja
    const backgroundColors = labelsBar.map(month =>
      month === filteredMonth ? "#fe6b13" : "#032740"
    );

    // ✅ Destruir gráficos anteriores antes de renderizar nuevos
    [pieChartRef, barChartRef, lineChartRef].forEach(ref => {
      if (ref.current?.chart) ref.current.chart.destroy();
    });

    // ✅ Gráfico de pastel - Tipos de avería
    pieChartRef.current.chart = new Chart(pieChartRef.current.getContext("2d"), {
      type: "pie",
      data: {
        labels: Object.keys(alertData.reduce((acc, alert) => {
          acc[alert.Tipo_avería || "Desconocido"] = (acc[alert.Tipo_avería || "Desconocido"] || 0) + 1;
          return acc;
        }, {})),
        datasets: [
          {
            label: "Tipos de Avería",
            data: Object.values(alertData.reduce((acc, alert) => {
              acc[alert.Tipo_avería || "Desconocido"] = (acc[alert.Tipo_avería || "Desconocido"] || 0) + 1;
              return acc;
            }, {})),
            backgroundColor: ["#fe6b13", "#032740", '#1e4e6c', '#5da3ea', '#bcd9f4', '#ffa76b', "#8E44AD"],
          },
        ],
      },
      options: { responsive: true, plugins: { title: { display: true, text: "Distribución de Tipos de Avería" } } },
    });

    // ✅ Gráfico de barras - Averías por mes
    barChartRef.current.chart = new Chart(barChartRef.current.getContext("2d"), {
      type: "bar",
      data: {
        labels: labelsBar,
        datasets: [
          {
            label: "Averías por Mes",
            data: valuesBar,
            backgroundColor: backgroundColors,
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: { beginAtZero: true },
        },
        plugins: {
          legend: { position: "top" },
          title: { display: true, text: "Averías por Mes" },
        },
      },
    });



  }, [alertData, fullAlertData, filteredMonth]);

  return (
    <>
      <div style={{ height: "40px" }}></div> {/* Espacio entre AlertTable y Graphs */}

      <div className="graph-container">
        <h2 className="graph-title">Estadísticas de Alertas</h2>
        <div className="chart-wrapper">
          <div className="graph-card"><canvas ref={pieChartRef}></canvas></div>
          <div className="graph-card"><canvas ref={barChartRef}></canvas></div>
        </div>
      </div>
    </>
  );

}

export default Graphs;
