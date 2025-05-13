### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.ndimage import gaussian_filter1d
#import datetime
# Suporte
import os
import sys
import joblib
# Pré-Processamento e Validações
import pymannkendall as mk

##### Padrão ANSI ###############################################################
bold = "\033[1m"
red = "\033[91m"
green = "\033[92m"
yellow = "\033[33m"
blue = "\033[34m"
magenta = "\033[35m"
cyan = "\033[36m"
white = "\033[37m"
reset = "\033[0m"
#################################################################################

_RETROAGIR = 7 # Dias

cidade = "Porto Alegre"
cidades = ["Porto Alegre"]
_CIDADE = cidade.upper()
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
for velho, novo in troca.items():
	_cidade = _CIDADE.replace(velho, novo)
	_cidade = _cidade.replace(" ", "_")
print(_cidade)
#sys.exit()

#########################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
verao = "biometeoro_verao_PORTO_ALEGRE.csv"
inverno = "biometeoro_inverno_PORTO_ALEGRE.csv"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
clima = "climatologia.csv"
biometeoro = "biometeoro_PORTO_ALEGRE.csv"
onda3 = "onda_calor_obito_3.3.csv"
onda5 = "onda_calor_obito_5.5.csv"

### Abrindo e Visualizando Arquivos
# Série Histórica de Óbitos Cardiovasculares
biometeoro = pd.read_csv(f"{caminho_dados}{biometeoro}", low_memory = False)
biometeoro["data"] = pd.to_datetime(biometeoro["data"])
print(f"\n{green}biometeoro\n{reset}{biometeoro}\n")
# Série Histórica de Ondas de Calor
onda3 = pd.read_csv(f"{caminho_dados}{onda3}", low_memory = False)
onda3["data"] = pd.to_datetime(onda3["data"])
# Série Histórica de Ondas de Calor
onda5 = pd.read_csv(f"{caminho_dados}{onda5}", low_memory = False)
onda5["data"] = pd.to_datetime(onda5["data"])
print(f"\n{green}onda 3.3\n{reset}{onda3}\n")
print(f"\n{green}onda 5.5\n{reset}{onda5}\n")

print(f"\n{green}Menores TMAX 3.3\n{reset}{onda3.where(onda3['tmax'] <= 25).dropna()}\n")
print(f"\n{green}onda de calor (acima 3 ºC, ao menos 3 dias)\n{reset}{onda3}\n")
print(f"\n{green}Menores TMAX 5.5\n{reset}{onda5.where(onda5['tmax'] <= 25).dropna()}\n")
print(f"\n{green}onda de calor (acima 5 ºC, ao menos 5 dias)\n{reset}{onda5}\n")
# VERÃO
verao = pd.read_csv(f"{caminho_dados}{verao}", low_memory = False)
verao["data"] = pd.to_datetime(verao["data"])
verao = verao.copy()
print(f"\n{green}PERÍODO SELECIONADO: VERÃO (DJF)\n{reset}{verao}\n{verao.info()}\n{verao.describe()}\n")
# INVERNO
inverno = pd.read_csv(f"{caminho_dados}{inverno}", low_memory = False)
inverno["data"] = pd.to_datetime(inverno["data"])
inverno = inverno.copy()
print(f"\n{green}PERÍODO SELECIONADO: INVERNO (JJA)\n{reset}{inverno}\n{inverno.info()}\n{inverno.describe()}\n")

### Estatística Descritiva
print(f"\n{green}Onda de calor (acima 3 ºC, ao menos 3 dias)\n{reset}{onda3.describe()}\n")
print(f"\n{green}Onda de calor (acima 5 ºC, ao menos 5 dias)\n{reset}{onda5.describe()}\n")
print(f"\n{green}Biometeorologia\n{reset}{biometeoro.describe()}\n")

### Visualizando Ondas de Calor
# Básico
"""
biometeoro[["tmax", "obito"]].plot()
plt.show()
sys.exit()
"""
# Com Ondas de Calor
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)

sns.lineplot(x = biometeoro["data"], y = biometeoro["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.5 )
sns.lineplot(x = biometeoro["data"], y = biometeoro["tmax"], label = "Temperatura Máxima",
				color = "red", linewidth = 1, alpha = 0.7 )
sns.scatterplot(x = onda3["data"], y = onda3["obito"],
				color = "blue", marker = "o",# alpha = 0.7,
				label = "Óbito em Onda de Calor (acima 3 ºC, ao menos 3 dias)")
sns.scatterplot(x = onda3["data"], y = onda3["tmax_clima"],
				color = "blue", marker = "o",# alpha = 0.7,
				label = "Temperatura em Onda de Calor (acima 3 ºC, ao menos 3 dias)")
sns.scatterplot(x = onda5["data"], y = onda5["obito"], #scatter
				color = "purple", marker = "D",# alpha = 0.7,
				label = "Óbito em Onda de Calor (acima 5 ºC, ao menos 5 dias)")
sns.scatterplot(x = onda5["data"], y = onda5["tmax_clima"],
				color = "purple", marker = "D",# alpha = 0.7,
				label = "Temperatura em Onda de Calor (acima 5 ºC, ao menos 5 dias)")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ONDAS DE CALOR.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares x Temperatura Máxima")
plt.show()
sys.exit()

if _SALVAR == "True":
	caminho_onda = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/ondacalor/"
	os.makedirs(caminho_onda, exist_ok = True)
	plt.savefig(f'{caminho_indice}onda_calor_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM1_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e ondas de calor do município de Porto Alegre.{reset}")
	plt.show()






