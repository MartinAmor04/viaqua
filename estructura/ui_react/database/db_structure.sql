USE curuxia_project;

-- Crear tabla de máquinas
CREATE TABLE machine (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    public_id VARCHAR(50) UNIQUE NOT NULL, 
    place VARCHAR(100) NOT NULL, 
    machine_type VARCHAR(100) NOT NULL, 
    power INT NOT NULL
);

-- Crear tabla de alertas con clave foránea y eliminación en cascada
CREATE TABLE alert (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    machine_id INT NOT NULL, 
    date_time DATETIME NOT NULL, 
    alert_type VARCHAR(100) NOT NULL, 
    audio_record LONGTEXT NOT NULL, 
    estado VARCHAR(50) DEFAULT 'Pendiente' NOT NULL,
    FOREIGN KEY (machine_id) REFERENCES machine(id) ON DELETE CASCADE
);
