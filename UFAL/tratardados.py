import pandas as pd

# Caminho para o arquivo CSV (usar \\ ou / ou string de caminho)
caminhoDados = "C:\\Users\\USUARIO(A)\\Documents\\GitHub\\adaptative-traffic-lights\\ColetaContinua\\2025\\AL\\104\\104_AL_km89_2025.csv"
df = pd.read_csv(caminhoDados)

print(df)