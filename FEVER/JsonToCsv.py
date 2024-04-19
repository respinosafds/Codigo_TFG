import json
import csv

archivo_json = "dataset.json"
archivo_csv = "dataset.csv"

# Lee el archivo JSON
with open(archivo_json, 'r') as f:
    datos_json = json.load(f)

# Abre el archivo CSV en modo de escritura
with open(archivo_csv, 'w', newline='', encoding='utf-8') as f:
    # Crea un objeto escritor CSV
    escritor_csv = csv.writer(f)

    # Escribe el encabezado utilizando las claves del primer objeto JSON
    escritor_csv.writerow(datos_json[0].keys())

    # Escribe cada fila del JSON como una fila en el CSV
    for fila in datos_json:
        escritor_csv.writerow(fila.values())

print(f"El archivo JSON '{archivo_json}' se ha convertido a '{archivo_csv}'.")