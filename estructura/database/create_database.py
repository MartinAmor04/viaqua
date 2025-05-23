import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv() 

MYSQL_HOST= os.getenv("DB_HOST")
MYSQL_USER= os.getenv("DB_USER")
MYSQL_PASSWORD= os.getenv("DB_PASSWORD")
MYSQL_NAME= os.getenv("DB_NAME")

conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_NAME
)
cursor = conn.cursor()

script_files = ["db_structure.sql"]  

for script_file in script_files:
    with open(script_file, "r") as file:
        sql_script = file.read()
        for statement in sql_script.split(";"):
            if statement.strip():
                cursor.execute(statement)

conn.commit()
cursor.close()
conn.close()

print("✅ Script ejecutado exitosamente!")
