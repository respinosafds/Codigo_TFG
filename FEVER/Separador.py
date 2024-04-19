import pandas as pd

# Especifica el nombre del archivo CSV y la columna para la separación
archivo_csv = 'BuenDataset.csv'
columna_separadora = 'verifiable'

# Lee el archivo CSV
datos = pd.read_csv(archivo_csv)

# Filtra los datos basándote en la columna especificada
grupo_1 = datos[datos[columna_separadora] == 'VERIFIABLE']
grupo_2 = datos[datos[columna_separadora] == 'NOT VERIFIABLE']

# Guarda los resultados en dos archivos CSV separados
grupo_1.to_csv('Real.csv', index=False)
grupo_2.to_csv('Falso.csv', index=False)

print("Archivos CSV separados con éxito.")