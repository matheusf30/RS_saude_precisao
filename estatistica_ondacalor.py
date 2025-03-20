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
"""
### CONDIÇÕES PARA VARIAR ########################################
##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZAR = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################
##################################################################
"""   
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

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteoro_porto_alegre.csv"
clima = "climatologia.csv"
#anomalia = "anomalia.csv"
#bio = "obito_cardiovasc_total_poa_96-22.csv"
bio = "obito_total_PORTO_ALEGRE.csv"
onda = "onda_calor_obito.csv"
#p75 = "serie_IAM3_porto_alegre.csv"

### Abrindo e Visualizando Arquivos
# Série Histórica Meteorológica
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", low_memory = False)
meteoro["data"] = pd.to_datetime(meteoro["data"])
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
# Série Histórica de Óbitos Cardiovasculares
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
bio["data"] = pd.to_datetime(bio["data"])
print(f"\n{green}bio\n{reset}{bio}\n")
# Série Histórica de Ondas de Calor
onda = pd.read_csv(f"{caminho_dados}{onda}", low_memory = False)
onda["data"] = pd.to_datetime(onda["data"])
#onda["acima"] = onda["tmax"] - onda ["tmax_clima"]
print(f"\n{green}Menores TMAX\n{reset}{onda.where(onda['tmax'] <= 25).dropna()}\n")
print(f"\n{green}onda de calor\n{reset}{onda}\n")

### Concatenando Meteorologia e Óbitos
biometeoro = meteoro.merge(bio, how = "inner", on = "data")
biometeoro["data"] = pd.to_datetime(biometeoro["data"])
nome_arquivo = f"biometeoro_{_cidade}.csv"
biometeoro.to_csv(f"{caminho_dados}{nome_arquivo}",index = False)
print(f"\n{green}Salvo com Sucesso:\n{reset}{caminho_dados}{nome_arquivo}\n")
print(f"\n{green}biometeoro\n{reset}{biometeoro}\n")

### Estatística Descritiva
print(f"\n{green}Onda de calor\n{reset}{onda.describe()}\n")
print(f"\n{green}Biometeorologia\n{reset}{biometeoro.describe()}\n")
#onda[["tmax", "tmax_clima", "obitos"]].plot()
#plt.show()

inverno = onda.copy()
inverno["dia"] = inverno["data"]
inverno.set_index("dia", inplace = True)
inverno.index = pd.to_datetime(inverno.index)
inverno = inverno[(inverno.index.month >= 6) & (inverno.index.month <= 8)]
inverno.reset_index(inplace = True)
inverno.drop(columns = "dia", inplace = True)
print(f"\n{green}PERÍODO SELECIONADO: INVERNO (JJA)\n{reset}{inverno}\n{inverno.info()}\n{inverno.describe()}\n")
verao = onda.copy()
verao["dia"] = verao["data"]
verao.set_index("dia", inplace = True)
verao.index = pd.to_datetime(verao.index)
verao = verao[(verao.index.month == 12) | (verao.index.month <= 2)]
verao.reset_index(inplace = True)
verao.drop(columns = "dia", inplace = True)
print(f"\n{green}PERÍODO SELECIONADO: VERÃO (DJF)\n{reset}{verao}\n{verao.info()}\n{verao.describe()}\n")

# Visualizando Ondas de Calor
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
biometeoro[["tmax", "obitos"]].plot()
"""
sns.lineplot(y = biometeoro["obitos"], label = "Óbito", #x = biometeoro["data"]
				color = "black", linewidth = 1, alpha = 0.7 )
sns.lineplot(x = biometeoro["data"], y = biometeoro["tmax"], label = "Temperatura Máxima",
				color = "black", linewidth = 1, alpha = 0.7 )
"""
sns.scatterplot(x = verao["data"], y = verao["obitos"],
				color = "purple", marker = "o", alpha = 0.7,
				label = "Óbito em Onda de Calor")
sns.scatterplot(x = verao["data"], y = verao["tmax_clima"], #scatter
				color = "orange", marker = "o", alpha = 0.7,
				label = "Temperatura Esperada (Média da Temperatura Máxima Diária)")
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






