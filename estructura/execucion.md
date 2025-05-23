1. crear un .env na raíz do proxecto (copiar o que pasei)
2. crear un virtual env (.venv) e facerlle un pip install dos requirments
3. instalar mysql server e client e crear un usuario e unha contraseña. Se son diferentes aos do meu .env cambiar!!!
4. Co .venv activo executar (isto inserta datos de proba):  
    ```
    python3 create_database.py
    python3 insert_aler_encoded_data.py
    ```
5. Entrar na carpeta 'ui_curuxia' e correr:
    ```
    streamlit run main.py
    ```
    Agora xa se debería ver unha interfaz (algo pocha) con dúas filas insertadas e cousas varias e feas metidas polo medio
6. Para tema broker de mensajería, entrar na carpeta 'pub_sub':
    - No ordena local executar
        ```
        python3 subscriber.py
        ```
    - Para o publisher, meter na raspi o publisher.py, o .env, o requirments.: 

        ```
        scp -r example_file.py pi@ip_address:/home/pi/folder
        ```
        Logo crear un .venv onde se instalen os requirements.txt
    - Por último (asegúrate de que o subscriber.py estea correndo en local):
        ```python
        python3 publisher.py #chama directamente a unha función send_alert() pero realmente dende outro podese importar noutro archivo 'from publisher.py send_alert()' e borra send_alert do publisher.py para que non a chame dúas veces
        ```
    - Agora debería ir todo e debería aparecer un novo rexistro no teu stramlit


