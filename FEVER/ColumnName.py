import pandas as pd

# Especifica el nombre del archivo CSV y el nombre de la columna a cambiar
archivo_csv = 'dataset.csv'
columna_a_cambiar = 'claim'
nuevo_nombre_columna = 'text'

# Lee el archivo CSV
datos = pd.read_csv(archivo_csv)

# Cambia el nombre de la columna especificada
datos = datos.rename(columns={columna_a_cambiar: nuevo_nombre_columna})

# Guarda los datos actualizados en un nuevo archivo CSV
datos.to_csv('BuenDataset.csv', index=False)

print(f"Nombre de columna cambiado de '{columna_a_cambiar}' a '{nuevo_nombre_columna}'. Nuevo archivo CSV creado: 'BuenDataset.csv'")