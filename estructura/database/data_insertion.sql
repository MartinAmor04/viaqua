INSERT INTO machine (public_id, place, machine_type, power) VALUES 
('MACH001', 'Vigo', 'Soplante', 500);

INSERT INTO alert (machine_id, date_time, alert_type, audio_record) VALUES
(1, NOW(), 'No clasificado', NULL);
