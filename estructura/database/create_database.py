import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="ines",
    password="miclave",
    database="curuxia_project"
)
cursor = conn.cursor()

script_files = ["db_structure.sql", "data_insertion.sql"]  

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
