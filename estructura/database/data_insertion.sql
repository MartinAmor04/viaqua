-- Insertar máquinas
ALTER TABLE machine AUTO_INCREMENT = 1;

INSERT INTO machine (public_id, place, machine_type, power) VALUES 
('SOP001', 'Sistís', 'Soplante', 10),
('SOP002', 'Santa Cruz 1 ', 'Soplante', 12),
('SOP003', 'Santa Cruz 2', 'Soplante', 15),
('SOP004', 'Casar do Mato', 'Soplante', 18),
('BOM005', 'Sistís', 'Bomba', 9),
('COM006', 'Santa Cruz 1', 'Compresor', 8),
('COM007', 'Santa Cruz 2', 'Compresor', 12),
('GEN008', 'Sistís', 'Motor', 20),
('MOT009', 'Casar do Mato', 'Motor', 6),
('BOM010', 'Casar do Mato', 'Bomba', 15);
