const API_URL = "http://127.0.0.1:5000/api";

export async function fetchAlerts(estado = "Pendiente") {
    try {
        const response = await fetch(`${API_URL}/alerts?estado=${estado}`);
        const data = await response.json();

        console.log("Datos de la API en fetchAlerts:", data); // Verifica si la API devuelve datos

        return data;
    } catch (error) {
        console.error("❌ Error al obtener alertas:", error);
        return [];
    }
}


export async function editAlert(id, alertType, estado) {
    try {
        const response = await fetch(`${API_URL}/edit-alert`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id, alert_type: alertType, estado }),
        });
        return await response.json();
    } catch (error) {
        console.error("❌ Error al editar alerta:", error);
        return { success: false };
    }
}

export async function getSoundFile(filename) {
    try {
        const response = await fetch(`${API_URL}/read-alert-file?filename=${filename}`);

        return await response.json();
    } catch (error) {
        console.error("❌ Error al obtener archivo:", error);
        return { success: false };
    }
}
