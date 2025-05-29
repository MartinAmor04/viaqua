2. Crear el entorno virtual de Python
python -m venv .venv
source .venv/bin/activate  
sudo apt update && sudo apt install pkg-config libmysqlclient-dev
pip install mysqlclient
pip install -r requirements.txt

Instalar MySQL y configurar usuario
sudo apt update && sudo apt install mysql-server mysql-client

2. Crear la base de datos automáticamente
sudo mysql
CREATE USER 'curuxia_admin'@'localhost' IDENTIFIED BY 'clave_segura';
GRANT ALL PRIVILEGES ON curuxia_project.* TO 'curuxia_admin'@'localhost';
FLUSH PRIVILEGES;
exit;

crear bbdd
python database/create_database.py 

 Ejecutar Flask
python backend/app.py 
hacer : (se debería ver json con datos)
curl http://127.0.0.1:5000/api/alerts

ejecutar front 

cd web/
npm install
npm install  leaflet
npm start

