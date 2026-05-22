import pandas as pd
from Libreria_de_Análisis_Estadístico import *

data = [
    {"Genero": "M", "Puntaje": 250},
    {"Genero": "F", "Puntaje": 300},
    {"Genero": "M", "Puntaje": 280},
    {"Genero": "M", "Puntaje": 250}

]

df = pd.DataFrame(data)

# Cualitativa
tabla, moda = frecuencia_cualitativa_manual(df["Genero"])
print(tabla)
print(moda)
# Cuantitativa
res = estadisticas_cuantitativas_manual(df["Puntaje"])
print(res)

tabla, duplicados, total_filas = diagnostico_manual(df)


print (duplicados)
