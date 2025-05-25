-- Insertar máquinas
ALTER TABLE machine AUTO_INCREMENT = 1;

INSERT INTO machine (public_id, place, machine_type, power) VALUES 
('MACH001', 'Vigo', 'Soplante', 500),
('MACH002', 'Madrid', 'Compresor', 750),
('MACH003', 'Barcelona', 'Generador', 1200),
('MACH004', 'Sevilla', 'Motor', 900),
('MACH005', 'Bilbao', 'Bomba', 600);

-- Insertar alertas con los estados distribuidos correctamente
INSERT INTO alert (machine_id, date_time, alert_type, audio_record, estado) VALUES
(1, '2025-05-01 08:30:00', 'Fallo eléctrico', 'BASE64_AUDIO_001', 'Pendiente'),
(2, '2025-05-02 10:15:00', 'Sobrecalentamiento', 'BASE64_AUDIO_002', 'Pendiente'),
(3, '2025-05-03 14:45:00', 'Pérdida de potencia', 'BASE64_AUDIO_003', 'Pendiente'),
(4, '2025-05-04 12:10:00', 'Fallo mecánico', 'BASE64_AUDIO_004', 'Pendiente'),
(5, '2025-05-05 18:25:00', 'Fallo en sensor', 'BASE64_AUDIO_005', 'En revisión'),
(1, '2025-05-06 09:00:00', 'Fallo eléctrico', 'BASE64_AUDIO_006', 'En revisión'),
(2, '2025-05-07 16:30:00', 'Pérdida de potencia', 'BASE64_AUDIO_007', 'Arreglada'),
(3, '2025-05-08 11:50:00', 'Sobrecalentamiento', 'BASE64_AUDIO_008', 'Arreglada'),
(4, '2025-05-09 07:40:00', 'Fallo mecánico', 'BASE64_AUDIO_009', 'Arreglada'),
(5, '2025-05-10 20:05:00', 'Fallo en sensor', 'BASE64_AUDIO_010', 'Arreglada');
