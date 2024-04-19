import pandas as pd

def eliminar_n_primeras_filas(archivo_entrada, archivo_salida, n_filas_a_eliminar):
    # Lee el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(archivo_entrada)

    # Elimina las primeras n filas
    df = df.iloc[n_filas_a_eliminar:]

    # Guarda el DataFrame resultante en un nuevo archivo CSV
    df.to_csv(archivo_salida, index=False)

# Especifica el nombre del archivo de entrada y salida, y la cantidad de filas a eliminar
archivo_entrada = 'Rial.csv'
archivo_salida = 'Riial.csv'
n_filas_a_eliminar = 14000  # Reemplaza con la cantidad real de filas que deseas eliminar

# Llama a la función para eliminar las primeras n filas
eliminar_n_primeras_filas(archivo_entrada, archivo_salida, n_filas_a_eliminar)