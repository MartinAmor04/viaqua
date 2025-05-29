-- Resetear AutoIncrement
-- Resetear IDs automáticos
ALTER TABLE machine AUTO_INCREMENT = 1;
ALTER TABLE alert AUTO_INCREMENT = 1;

-- Insertar máquinas en ubicaciones de Galicia
INSERT INTO machine (public_id, place, machine_type, power) VALUES 
('MACH001', 'Vigo', 'Soplante', 500),
('MACH002', 'A Coruña', 'Compresor', 750),
('MACH003', 'Santiago de Compostela', 'Generador', 1200),
('MACH004', 'Lugo', 'Motor', 900),
('MACH005', 'Ourense', 'Bomba', 600);

-- Insertar alertas durante todo el año (cada mes)
INSERT INTO alert (machine_id, date_time, alert_type, audio_record, estado) VALUES
-- Enero
(1, '2025-01-05 08:30:00', 'Fallo eléctrico', 'BASE64_AUDIO_001', 'Pendiente'),
(2, '2025-01-12 10:15:00', 'Sobrecalentamiento', 'BASE64_AUDIO_002', 'Pendiente'),
(3, '2025-01-20 14:45:00', 'Pérdida de potencia', 'BASE64_AUDIO_003', 'Pendiente'),

-- Febrero
(4, '2025-02-03 12:10:00', 'Fallo mecánico', 'BASE64_AUDIO_004', 'Pendiente'),
(5, '2025-02-15 18:25:00', 'Fallo en sensor', 'BASE64_AUDIO_005', 'En revisión'),
(1, '2025-02-28 09:00:00', 'Fallo eléctrico', 'BASE64_AUDIO_006', 'En revisión'),

-- Marzo
(2, '2025-03-07 16:30:00', 'Pérdida de potencia', 'BASE64_AUDIO_007', 'Arreglada'),
(3, '2025-03-18 11:50:00', 'Sobrecalentamiento', 'BASE64_AUDIO_008', 'Arreglada'),

-- Abril
(4, '2025-04-02 07:40:00', 'Fallo mecánico', 'BASE64_AUDIO_009', 'Arreglada'),
(5, '2025-04-27 20:05:00', 'Fallo en sensor', 'BASE64_AUDIO_010', 'Arreglada'),

-- Mayo
(1, '2025-05-01 08:30:00', 'Fallo eléctrico', 'BASE64_AUDIO_011', 'Pendiente'),
(2, '2025-05-10 10:15:00', 'Sobrecalentamiento', 'BASE64_AUDIO_012', 'Pendiente'),

-- Junio
(3, '2025-06-15 14:45:00', 'Pérdida de potencia', 'BASE64_AUDIO_013', 'Pendiente'),
(4, '2025-06-22 12:10:00', 'Fallo mecánico', 'BASE64_AUDIO_014', 'Pendiente'),

-- Julio
(5, '2025-07-03 18:25:00', 'Fallo en sensor', 'BASE64_AUDIO_015', 'En revisión'),
(1, '2025-07-19 09:00:00', 'Fallo eléctrico', 'BASE64_AUDIO_016', 'En revisión'),

-- Agosto
(2, '2025-08-05 16:30:00', 'Pérdida de potencia', 'BASE64_AUDIO_017', 'Arreglada'),
(3, '2025-08-11 11:50:00', 'Sobrecalentamiento', 'BASE64_AUDIO_018', 'Arreglada'),

-- Septiembre
(4, '2025-09-07 07:40:00', 'Fallo mecánico', 'BASE64_AUDIO_019', 'Arreglada'),
(5, '2025-09-29 20:05:00', 'Fallo en sensor', 'BASE64_AUDIO_020', 'Arreglada'),

-- Octubre
(1, '2025-10-02 08:30:00', 'Fallo eléctrico', 'BASE64_AUDIO_021', 'Pendiente'),
(2, '2025-10-18 10:15:00', 'Sobrecalentamiento', 'BASE64_AUDIO_022', 'Pendiente'),

-- Noviembre
(3, '2025-11-12 14:45:00', 'Pérdida de potencia', 'BASE64_AUDIO_023', 'Pendiente'),
(4, '2025-11-23 12:10:00', 'Fallo mecánico', 'BASE64_AUDIO_024', 'Pendiente'),

-- Diciembre
(5, '2025-12-05 18:25:00', 'Fallo en sensor', 'BASE64_AUDIO_025', 'En revisión'),
(1, '2025-12-28 09:00:00', 'Fallo eléctrico', 'BASE64_AUDIO_026', 'En revisión');
