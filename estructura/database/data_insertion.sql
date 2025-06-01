-- Insertar máquinas
ALTER TABLE machine AUTO_INCREMENT = 1;

INSERT INTO machine (public_id, place, machine_type, power) VALUES 
('SOP001', 'Dodro', 'Soplante', 10),
('SOP002', 'Laxe  ', 'Soplante', 12),
('SOP003', 'Toén', 'Soplante', 15),
('SOP004', 'Vilaboa', 'Soplante', 18),
('BOM005', 'Dodro', 'Bomba', 9),
('COM006', 'Laxe ', 'Compresor', 8),
('COM007', 'Toén', 'Compresor', 12),
('GEN008', 'Dodro', 'Motor', 20),
('MOT009', 'Vilaboa', 'Motor', 6),
('BOM010', 'Vilaboa', 'Bomba', 15);
