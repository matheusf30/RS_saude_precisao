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
### Encaminhamento aos Diretórios
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
clima = "climatologia.csv"
biometeoro = "biometeoro_PORTO_ALEGRE.csv"

### Abrindo e Visualizando Arquivos
# Série Histórica Meteorológica com Óbitos
biometeoro = pd.read_csv(f"{caminho_dados}{biometeoro}", low_memory = False)
biometeoro["data"] = pd.to_datetime(biometeoro["data"])
print(f"\n{green}biometeoro\n{reset}{biometeoro}\n")
# Série Climatológica Diária
clima = pd.read_csv(f"{caminho_dados}{clima}", low_memory = False)
#clima["data"] = pd.to_datetime(clima["data"])
print(f"\n{green}clima\n{reset}{clima}\n")

### Selecionando Variáveis e Pré-processando
tmax = biometeoro[["data", "tmax"]]
tmax["dia"] = tmax["data"].dt.dayofyear
tmax_media = tmax.groupby("dia")["tmax"].mean().round(2)
tmax["tmax_clima"] = tmax["dia"].map(tmax_media)
print(f"\n{green}tmax\n{reset}{tmax}\n")
tmax_clima = clima[["dia", "tmax"]]
print(f"\n{green}tmax_clima\n{reset}{tmax_clima}\n")

### Processando Ondas de Calor (acima 5 graus por 5 dias - 5.5)
tmax["acima_5"] = tmax["tmax"] > (tmax["tmax_clima"] + 5)
print(f"\n{green}tmax\n{reset}{tmax}\n")
print(f"\n{green}Dias com temperatura máxima acima da média:\n{reset}{tmax['acima_5'].value_counts()}\n")
tmax["agrupado"] = (tmax["acima_5"] != tmax["acima_5"].shift()).cumsum()
print(f"\n{green}tmax\n{reset}{tmax}\n")
ondadecalor = tmax.groupby("agrupado").filter(lambda x: x["acima_5"].all() and len(x) >= 5)
print(f"\n{green}ondadecalor\n{reset}{ondadecalor}\n")
if not ondadecalor.empty:
	onda_calor = ondadecalor[["data", "tmax", "tmax_clima"]]
	print(f"\n{green}Onda de Calor:\n{reset}{onda_calor}\n", print(onda_calor.columns))
else:
	print(f"\n{red}Nenhuma Onda de Calor Detectada!\n{reset}")
onda_calor = onda_calor.copy()
onda_calor.loc[:, "acima"] = onda_calor["tmax"] - onda_calor["tmax_clima"]

### Concatenando Óbitos
#onda_calor["obito"] = onda_calor["data"].map(bio["obitos"])
onda_calor = onda_calor.merge(biometeoro, how = "inner", on = "data")
print(f"\n{green}onda_calor\n{reset}{onda_calor}\n")
onda_calor = onda_calor.drop(columns = "tmax_x")
nome_arquivo = "onda_calor_obito_5.5.csv"
onda_calor.rename(columns = {"tmax_y": "tmax"}, inplace = True)
print(f"\n{green}Onda de Calor:\n{reset}{onda_calor}\n", print(onda_calor.columns))
onda_calor.to_csv(f"{caminho_dados}{nome_arquivo}",index = False)
print(f"\n{green}Salvo com Sucesso:\n{reset}{caminho_dados}{nome_arquivo}\n")

### Selecionando Variáveis e Pré-processando
tmax = biometeoro[["data", "tmax"]]
tmax["dia"] = tmax["data"].dt.dayofyear
tmax_media = tmax.groupby("dia")["tmax"].mean().round(2)
tmax["tmax_clima"] = tmax["dia"].map(tmax_media)
print(f"\n{green}tmax\n{reset}{tmax}\n")
tmax_clima = clima[["dia", "tmax"]]
print(f"\n{green}tmax_clima\n{reset}{tmax_clima}\n")

### Processando Ondas de Calor (acima 3 graus por 3 dias - 3.3)
tmax["acima_3"] = tmax["tmax"] > (tmax["tmax_clima"] + 3)
print(f"\n{green}tmax\n{reset}{tmax}\n")
print(f"\n{green}Dias com temperatura máxima acima da média:\n{reset}{tmax['acima_3'].value_counts()}\n")
tmax["agrupado"] = (tmax["acima_3"] != tmax["acima_3"].shift()).cumsum()
print(f"\n{green}tmax\n{reset}{tmax}\n")
ondadecalor = tmax.groupby("agrupado").filter(lambda x: x["acima_3"].all() and len(x) >= 3)
print(f"\n{green}ondadecalor\n{reset}{ondadecalor}\n")
if not ondadecalor.empty:
	onda_calor = ondadecalor[["data", "tmax", "tmax_clima"]]
	print(f"\n{green}Onda de Calor:\n{reset}{onda_calor}\n", print(onda_calor.columns))
else:
	print(f"\n{red}Nenhuma Onda de Calor Detectada!\n{reset}")
onda_calor = onda_calor.copy()
onda_calor.loc[:, "acima"] = onda_calor["tmax"] - onda_calor["tmax_clima"]

### Concatenando Óbitos
onda_calor = onda_calor.merge(biometeoro, how = "inner", on = "data")
onda_calor = onda_calor.drop(columns = "tmax_x")
print(f"\n{green}onda_calor\n{reset}{onda_calor}\n")
nome_arquivo = "onda_calor_obito_3.3.csv"
onda_calor.rename(columns = {"tmax_y": "tmax"}, inplace = True)
print(f"\n{green}Onda de Calor:\n{reset}{onda_calor}\n", print(onda_calor.columns))
onda_calor.to_csv(f"{caminho_dados}{nome_arquivo}",index = False)
print(f"\n{green}Salvo com Sucesso:\n{reset}{caminho_dados}{nome_arquivo}\n")
