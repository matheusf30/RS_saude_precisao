### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
#import datetime
# Suporte
import os
import sys
import joblib
# Pré-Processamento e Validações
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score#, RocCurveDisplay
from sklearn.inspection import permutation_importance
# Modelos e Visualizações
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import export_graphviz, export_text, plot_tree
#from sklearn.utils.graph import single_source_shortest_path_lenght as short_path

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

### CONDIÇÕES PARA VARIAR ##################################
                                                       #####
_LOCAL = "SIFAPSC" # OPÇÕES >>> GH|CASA|CLUSTER|SIFAPSC#####
                                                       #####
##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################
        
_RETROAGIR = 7 # Dias
_HORIZONTE = 0 # Tempo de Previsão
_JANELA_MM = 0 # Média Móvel
_K = 0 # constante para formar MM

cidade = "Porto Alegre"
cidade = cidade.upper()
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
for velho, novo in troca.items():
	_cidade = cidade.replace(velho, novo)
	_cidade = _cidade.replace(" ", "_")
print(_cidade)
#sys.exit()

#########################################################################

### Encaminhamento aos Diretórios
if _LOCAL == "GH":
	caminho_dados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/dados/"
	caminho_resultados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/resultados/porto_alegre/"
	caminho_modelos = "https://github.com/matheusf30/RS_saude_precisao/tree/main/modelos/"
elif _LOCAL == "SIFAPSC":
	caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
	caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
	caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"
elif _LOCAL == "CLUSTER":
	caminho_dados = "..."
elif _LOCAL == "CASA":
	caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/"
	caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/modelos/"
else:
	print("CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteo_poa_h_96-22.csv"
bio = "obito_cardiovasc_total_poa_96-22.csv"
oc = "PoA_oc_1996-01-01_2022-12-31.xlsx"
"""
prec = "prec_semana_ate_2023.csv"
tmin = "tmin_semana_ate_2023.csv"
tmed = "tmed_semana_ate_2023.csv"
tmax = "tmax_semana_ate_2023.csv"
"""
### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", skiprows = 10, sep = ";", low_memory = False)
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
oc = 
"""
prec = pd.read_csv(f"{caminho_dados}{prec}")
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
"""



### Pré-Processamento

# BIO-SAÚDE
bio.rename(columns = {"CAUSABAS" : "causa"}, inplace = True)
bio["data"] = pd.to_datetime(bio[["anoobito", "mesobito", "diaobito"]].astype(str).agg("-".join, axis = 1), format = "%Y-%m-%d")
bio.reset_index(inplace = True)
bio["obito"] = np.ones(len(bio)).astype(int)
bio.drop(columns=["CODMUNRES", "diaobito", "mesobito", "anoobito"], inplace = True)
bio = bio[["data", "obito", "sexo", "idade", "causa"]].sort_values(by = "data")
#bio = bio.groupby(by = ["data"])["obito"].sum()
total = bio.groupby(by = ["data"])["obito"].sum()
#sexo = bio.groupby(by = ["data", "sexo"])["obito"].sum()
#idade = bio.groupby(by = ["data", "idade"])["obito"].sum()
#causa = bio.groupby(by = ["data", "causa"])["obito"].sum()

# METEOROLOGIA
meteoro.rename(columns = {"Data Medicao" : "data",
						"Hora Medicao" : "hora",
						"PRECIPITACAO TOTAL, HORARIO(mm)" : "prec",
						"PRESSAO ATMOSFERICA AO NIVEL DO MAR, HORARIA(mB)" : "pressao",
						"TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)" : "temp",
						"UMIDADE RELATIVA DO AR, HORARIA(%)" : "umidade",
						"VENTO, DIRECAO HORARIA(codigo)" : "ventodir",
						"VENTO, VELOCIDADE HORARIA(m/s)" : "ventovel"}, inplace = True)
meteoro.drop(columns = "Unnamed: 8", inplace = True)
meteoro.dropna(axis = 0, inplace = True)
colunas_objt = meteoro.select_dtypes(include='object').columns
meteoro = meteoro.replace("," , ".")
meteoro[colunas_objt] = meteoro[colunas_objt].apply(lambda x: x.str.replace(",", "."))
meteoro["prec"] = pd.to_numeric(meteoro["prec"], errors = "coerce")
meteoro["pressao"] = pd.to_numeric(meteoro["pressao"], errors = "coerce")
meteoro["temp"] = pd.to_numeric(meteoro["temp"], errors = "coerce")
meteoro["ventovel"] = pd.to_numeric(meteoro["ventovel"], errors = "coerce")
meteoro = meteoro.groupby("data").filter(lambda x: len(x) == 3)
print(meteoro)
#sys.exit()
prec = meteoro.groupby(by = ["data"])["prec"].sum()
tmax = meteoro.groupby(by = ["data"])["temp"].max()
tmin = meteoro.groupby(by = ["data"])["temp"].min()
urmin = meteoro.groupby(by = ["data"])["umidade"].min()
urmax = meteoro.groupby(by = ["data"])["umidade"].max()
meteoro = meteoro.groupby(by = "data")[["pressao", "temp", "umidade", "ventodir", "ventovel"]].mean().round(2)
print(tmin)
meteoro["prec"] = prec
meteoro["tmin"] = tmin
meteoro["tmax"] = tmax
meteoro["urmin"] = urmin
meteoro["urmax"] = urmax
meteoro["amplitude_t"] = meteoro["tmax"] - meteoro["tmin"]
print(meteoro)
#sys.exit()

# BIOMETEORO
meteoro.reset_index(inplace = True)
meteoro["data"] = pd.to_datetime(meteoro["data"])
total = total.to_frame(name = "obito")
total.reset_index(inplace = True)
biometeoro = meteoro.merge(total, on = "data", how = "inner")
biometeoro = biometeoro[["data", "obito",
						"tmin", "temp", "tmax", "amplitude_t",
						"urmin", "umidade", "urmax",
						"prec", "pressao", "ventodir", "ventovel"]]


print(80*"=")
print(bio, bio.info())
print(80*"=")
print(total, total.info())
print(80*"=")
print(prec, prec.info())
print(80*"=")
print(meteoro, meteoro.info())
print(80*"=")
print(biometeoro, biometeoro.info())

#sys.exit()

# Visualização Prévia
_sigma = int(input(f"\n{ansi['cyan']}>>> Caso a suavização não seja necessária, digite zero (0).\n>>> Caso seja, selecione um número inteiro maior que zero: {ansi['reset']}"))
if _sigma > 0:
	biometeoro["s_obito"] = biometeoro["obito"].copy()
	biometeoro["s_obito"] = gaussian_filter1d(biometeoro["s_obito"],
		                                   sigma = _sigma)
	biometeoro["s_temp"] = biometeoro["temp"].copy()
	biometeoro["s_temp"] = gaussian_filter1d(biometeoro["s_temp"],
		                                   sigma = _sigma*5)
	biometeoro["s_tmin"] = biometeoro["tmin"].copy()
	biometeoro["s_tmin"] = gaussian_filter1d(biometeoro["s_tmin"],
		                                   sigma = _sigma*5)
	biometeoro["s_tmax"] = biometeoro["tmax"].copy()
	biometeoro["s_tmax"] = gaussian_filter1d(biometeoro["s_tmax"],
		                                   sigma = _sigma*5)
	biometeoro["s_amplitude_t"] = biometeoro["amplitude_t"].copy()
	biometeoro["s_amplitude_t"] = gaussian_filter1d(biometeoro["s_amplitude_t"],
		                                   sigma = _sigma*5)
	plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
	ax1 = plt.gca()
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_temp"], zorder = 1,
		        color = "gold", alpha = 0.7, linewidth = 1,
				label = "Temperatura Média", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_tmin"], zorder = 1,
		        color = "blue", alpha = 0.7, linewidth = 1,
				label = "Temperatura Mínima", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_tmax"], zorder = 1,
		        color = "red", alpha = 0.7, linewidth = 1,
				label = "Temperatura Máxima", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_amplitude_t"], zorder = 2,
		        color = "green", alpha = 0.7, linewidth = 1,
				label = "Amplitude Térmica", ax = ax1)
	ax2 = ax1.twinx()
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_obito"], zorder = 2,
		         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
	ax2.set_title(f"SUAVIZAÇÃO GAUSSIANA ($\\sigma$) DA SÉRIE HISTÓRICA, {cidade}, RIO GRANDE DO SUL.\nANÁLISE EXPLORATÓRIA DE TEMPERATURAS ($\\sigma$ = {_sigma*5}) E ÓBITOS CARDIOVASCULARES ($\\sigma$ = {_sigma}).")
else:
	plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
	ax1 = plt.gca()
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_temp"], zorder = 1,
		        color = "gold", alpha = 0.7, linewidth = 1,
				label = "Temperatura Média", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_tmin"], zorder = 1,
		        color = "blue", alpha = 0.7, linewidth = 1,
				label = "Temperatura Mínima", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_tmax"], zorder = 1,
		        color = "red", alpha = 0.7, linewidth = 1,
				label = "Temperatura Máxima", ax = ax1)
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_amplitude_t"], zorder = 2,
		        color = "green", alpha = 0.7, linewidth = 1,
				label = "Amplitude Térmica", ax = ax1)
	ax2 = ax1.twinx()
	sns.lineplot(x = biometeoro["data"], y = biometeoro["s_obito"], zorder = 2,
		         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
	ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA, {cidade}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
#sns.lineplot(x = biometeoro["data"], y = biometeoro["prec"],
#             color = "darkblue", alpha = 0.7, linewidth = 3, label = "Precipitação")
if _SALVAR == "True":
	plt.savefig(f'{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf', format = "pdf", dpi = 1200)
	print(f"{ansi['green']}SALVANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{ansi['reset']}")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}VISUALIZANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{ansi['reset']}")
	plt.show()


