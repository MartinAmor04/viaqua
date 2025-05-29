/**
 * Convierte los datos de alertas en un formato legible para la interfaz.
 * @param {Array} alerts - Lista de alertas desde la API
 * @returns {Array} - Lista formateada con datos listos para la UI
 */
export function formatAlerts(alerts) {
  return alerts.map(alert => ({
    id: alert.ID,
    machine: alert.Máquina,
    type: alert.Tipo,
    dateTime: new Date(alert.Fecha_hora).toLocaleString(),
    location: alert.Ubicación,
    issueType: alert.Tipo_avería,
    status: alert.Estado,
    audio: alert.Audio ? `data:audio/mp3;base64,${alert.Audio}` : null
  }));
}

/**
 * Filtra alertas por estado
 * @param {Array} alerts - Lista de alertas
 * @param {string} status - Estado deseado ("Pendiente", "En revisión", etc.)
 * @returns {Array} - Lista filtrada de alertas
 */
export function filterAlertsByStatus(alerts, status) {
  return alerts.filter(alert => alert.status === status);
}
