import json

archivo_jsonl = "train.jsonl"
archivo_json = "dataset.json"

# Lee el archivo JSONL y convierte cada línea a un diccionario Python
with open(archivo_jsonl, 'r') as f:
    lineas_jsonl = f.readlines()

# Combina las líneas en una única cadena JSON
contenido_json = "[" + ",".join(lineas_jsonl) + "]"

# Convierte la cadena JSON a un objeto Python
datos_json = json.loads(contenido_json)

# Escribe el objeto Python como un archivo JSON
with open(archivo_json, 'w') as f:
    json.dump(datos_json, f, indent=2)

print(f"El archivo JSONL '{archivo_jsonl}' se ha convertido a '{archivo_json}'.")