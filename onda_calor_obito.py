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
"""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score#, RocCurveDisplay
from sklearn.inspection import permutation_importance
matplotlib.use("Agg")
import shap
print(f"shap.__version__ = {shap.__version__}") #0.46.0
# Modelos e Visualizações
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import export_graphviz, export_text, plot_tree
#from sklearn.utils.graph import single_source_shortest_path_lenght as short_path
"""
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
#caminho_indices = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
#caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteoro_porto_alegre.csv"
clima = "climatologia.csv"
#anomalia = "anomalia.csv"
#bio = "obito_cardiovasc_total_poa_96-22.csv"
bio = "obito_total_PORTO_ALEGRE.csv"
#p75 = "serie_IAM3_porto_alegre.csv"

### Abrindo e Visualizando Arquivos
# Série Histórica Meteorológica
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", low_memory = False)
meteoro["data"] = pd.to_datetime(meteoro["data"])
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
# Série Climatológica Diária

clima = pd.read_csv(f"{caminho_dados}{clima}", low_memory = False)
#clima["data"] = pd.to_datetime(clima["data"])
print(f"\n{green}clima\n{reset}{clima}\n")
# Série Histórica Óbitos
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
bio["data"] = pd.to_datetime(bio["data"])
print(f"\n{green}bio\n{reset}{bio}\n")

### Selecionando Variáveis e Pré-processando
tmax = meteoro[["data", "tmax"]]
tmax["dia"] = tmax["data"].dt.dayofyear
tmax_media = tmax.groupby("dia")["tmax"].mean().round(2)
tmax["tmax_clima"] = tmax["dia"].map(tmax_media)
print(f"\n{green}tmax\n{reset}{tmax}\n")
tmax_clima = clima[["dia", "tmax"]]
print(f"\n{green}tmax_clima\n{reset}{tmax_clima}\n")

### Processando Ondas de Calor
tmax["acima_5"] = tmax["tmax"] > (tmax["tmax_clima"] + 5)
print(f"\n{green}tmax\n{reset}{tmax}\n")
print(f"\n{green}Dias com temperatura máxima acima da média:\n{reset}{tmax['acima_5'].value_counts()}\n")
tmax["agrupado"] = (tmax["acima_5"] != tmax["acima_5"].shift()).cumsum()
print(f"\n{green}tmax\n{reset}{tmax}\n")
ondadecalor = tmax.groupby("agrupado").filter(lambda x: x["acima_5"].all() and len(x) >= 5)
print(f"\n{green}ondadecalor\n{reset}{ondadecalor}\n")
if not ondadecalor.empty:
	onda_calor = ondadecalor[['data', 'tmax', 'tmax_clima']]
	print(f"\n{green}Onda de Calor:\n{reset}{onda_calor}\n")
else:
	print(f"\n{red}Nenhuma Onda de Calor Detectada!\n{reset}")
onda_calor = onda_calor.copy()
onda_calor.loc[:, "acima"] = onda_calor["tmax"] - onda_calor["tmax_clima"]

### Concatenando Óbitos
#onda_calor["obito"] = onda_calor["data"].map(bio["obitos"])
onda_calor = onda_calor.merge(bio, how = "inner", on = "data")
print(f"\n{green}onda_calor\n{reset}{onda_calor}\n")
if _SALVAR == True:
	nome_arquivo = "onda_calor_obito.csv"
	onda_calor.to_csv(f"{caminho_dados}{nome_arquivo}",index = False)
	print(f"\n{green}Salvo com Sucesso:\n{reset}{caminho_dados}{nome_arquivo}\n")
