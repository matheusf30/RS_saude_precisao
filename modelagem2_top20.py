### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
#import datetime
# Suporte
import os
import sys
import joblib
# Pré-Processamento e Validações
import pymannkendall as mk
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
_HORIZONTE = 0 # Tempo de Previsão
_JANELA_MM = 0 # Média Móvel
_K = 0 # constante para formar MM

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
caminho_indices = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
caminho_shap = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/SHAP/"
caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteoro_porto_alegre.csv"
clima = "climatologia.csv"
anomalia = "anomalia.csv"
bio = "obito_cardiovasc_total_poa_96-22.csv"
p75 = "serie_IAM3_porto_alegre.csv"


### Abrindo Arquivo e Pré-processamento
# Meteoro
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", low_memory = False)
meteoro["data"] = pd.to_datetime(meteoro["data"])
clima = pd.read_csv(f"{caminho_dados}{clima}", low_memory = False)
anomalia = pd.read_csv(f"{caminho_dados}{anomalia}", low_memory = False)
### ÍNDICES SIMPLIFICADOS DE SENSAÇÃO TÉRMICA
# Wind Chill #https://www.meteoswiss.admin.ch/weather/weather-and-climate-from-a-to-z/wind-chill.html
# Farenheit # miles per hour # https://www.weather.gov/media/epz/wxcalc/windChill.pdf
meteoro["ventovel"] = meteoro["ventovel"] * 3.6 # m/s >> km/h
Tc = meteoro["temp"] # T (C) < 10
Tf = ((meteoro["temp"] * 9) / 5) + 32  # C >> F 
Vkmh = meteoro["ventovel"] * 3.6 # ms >> km/h # V (km/h) > 4.82
Vmph = meteoro["ventovel"] * 2.237 # ms >> mph 
RH = meteoro["umidade"] # %
#meteoro["wind_chillC"] = 13.12 + 0.6215 * Tc - 11.37 * Vkmh**0.16 + 0.3965 * Tc * Vkmh**0.16 # C km/h # Fazer condicionantes
meteoro["wind_chill"] = 35.74 + 0.6215 * Tf - 35.75 * Vmph**0.16 + 0.4275 * Tf * Vmph**0.16 # F mph # Fazer condicionantes
#meteoro["wind_chill"] = np.where((Tc < 10) & (Vkmh > 4.82),  meteoro["wind_chill"], None)
# Heat Index #
meteoro["heat_index"] =  -42.379 + 2.04901523*Tf + 10.14333127*RH - .22475541*Tf*RH - .00683783*Tf*Tf - .05481717*RH*RH + .00122874*Tf*Tf*RH + .00085282*Tf*RH*RH - .00000199*Tf*Tf*RH*RH
ajuste1 = ((13 - RH) / 4) * np.sqrt((17 - np.absolute(Tf - 95.)) / 17)
ajuste2 =  ((RH - 85) / 10) * ((87 - Tf) / 5)
#meteoro["heat_index"] = np.where((RH <= 13) & (Tf >= 80) & (Tf <= 112),  meteoro["heat_index"] - ajuste1, None)
#meteoro["heat_index"] = np.where((RH >= 85) & (Tf >= 80) & (Tf <= 87),  meteoro["heat_index"] - ajuste2, None)
### ÍNDICES DE AMPLITUDE TÉRMICA
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
meteoro["indice_amplitude_up"] = meteoro["amplitude_t"] ** 2
meteoro["indice_amplitude_down"] = (1 / meteoro["amplitude_t"]) ** 2
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
### ANOMALIAS ESTACIONÁRIAS
sazonal = meteoro.copy()
sazonal["data"] = pd.to_datetime(sazonal["data"]).dt.date
sazonal = sazonal.astype({"data": "datetime64[ns]"})
sazonal["dia"] = sazonal["data"].dt.dayofyear
print(f"\n{green}sazonal\n{reset}{sazonal}\n")
colunas_sazonal = sazonal.drop(columns = "data")
colunas = colunas_sazonal.columns
media_dia = sazonal.groupby("dia")[colunas].mean().round(2)
media_dia.to_csv(f"{caminho_dados}climatologia.csv", index = True)
print(f"\n{green}SALVO COM SUCESSO!\nCLIMATOLOGIA.csv\n{reset}")
#media_dia.reset_index(inplace = True)
print(f"\n{green}media_dia\n{reset}{media_dia}\n{green}media_dia.index\n{reset}{media_dia.index}")
print(f"\n{green}sazonal\n{reset}{sazonal}\n")
componente_sazonal = sazonal.merge(media_dia, left_on = "dia", how = "left", suffixes = ("", "_media"), right_index = True)
sem_sazonal = pd.DataFrame(index = sazonal.index)
dias = componente_sazonal["dia"]
componente_sazonal.drop(columns = "dia", inplace = True)
print(f"{green}componente_sazonal\n{reset}{componente_sazonal}")
for coluna in colunas:
		if coluna in componente_sazonal.columns:
			media_coluna = f"{coluna}_media"
			if media_coluna in componente_sazonal.columns:
				sem_sazonal[coluna] = sazonal[coluna] - componente_sazonal[media_coluna]
			else:
				print(f"{red}Coluna {media_coluna} não encontrada no componente sazonal!{reset}")
		else:
			print(f"{red}Coluna {coluna} não encontrada no csv!{reset}")
sem_sazonal["dia"] = dias
sem_sazonal.dropna(inplace = True)
print(f"\n{green}sem_sazonal\n{reset}{sem_sazonal}\n")
print(f"\n{green}sem_sazonal.columns\n{reset}{sem_sazonal.columns}\n")
sem_sazonal.to_csv(f"{caminho_dados}anomalia.csv", index = True)
print(f"\n{green}SALVO COM SUCESSO!\nANOMALIA.csv\n{reset}")
colunas_anomalia = sem_sazonal.drop(columns = ["dia"])
sem_sazonal.dropna(axis = 1, inplace = True)
colunas = colunas_anomalia.columns
anomalia_estacionaria = pd.DataFrame()
anomalia_estacionaria["dia"] = dias
for c in colunas:
	if len(sem_sazonal[c]) > 1:
		tendencia = mk.original_test(sem_sazonal[c])
		print(f"{cyan}\nVARIÁVEL\n{c.upper()}{reset}\n")
		print(f"\n{green}sem_sazonal[c]\n{reset}{sem_sazonal[c]}\n")
		print(f"\n{green}tendencia\n{reset}{tendencia}\n")
		sem_tendencia = sem_sazonal[c] -(tendencia.slope + tendencia.intercept)# * len(sem_sazonal[c]))
		anomalia_estacionaria[c] = sem_tendencia
	else:
		print(f"{red}Coluna Faltante: {c}\nINSUFICIÊNCIA DE DADOS!\n(Tamanho: {len(sem_sazonal[c])}).{reset}")
print(f"\n{green}anomalia_estacionaria\n{reset}{anomalia_estacionaria}\n")
meteoro["dia"] = dias
meteoro_origem = meteoro.copy()
#meteoro = meteoro.merge(anomalia, left_on = "dia", how = "left", suffixes = ("", "_anomal"), right_index = True)
#meteoro.drop(columns = ["dia", "dia_anomal"], inplace = True)
meteoro = meteoro.merge(anomalia_estacionaria, left_on = "dia", how = "left", suffixes = ("", "_aest"), right_index = True)
meteoro.drop(columns = ["dia", "dia_aest"], inplace = True)
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
#sys.exit()
#
# Saúde
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
bio.rename(columns = {"CAUSABAS" : "causa"}, inplace = True)
bio["data"] = pd.to_datetime(bio[["anoobito", "mesobito", "diaobito"]].astype(str).agg("-".join, axis = 1), format = "%Y-%m-%d")
bio.reset_index(inplace = True)
bio["obito"] = np.ones(len(bio)).astype(int)
bio.drop(columns=["CODMUNRES", "diaobito", "mesobito", "anoobito"], inplace = True)
bio = bio[["data", "obito", "sexo", "idade", "causa"]].sort_values(by = "data")
print(f"\n{green}bio:\n{reset}{bio}\n")
obito_total = bio.groupby(by = ["data"])["obito"].sum()
obito_total = obito_total.reset_index()
obito_total.columns = ["data", "obitos"]
obito_total.to_csv(f"{caminho_dados}obito_total_{_cidade}.csv", index = False)
obito_agrupado = bio.groupby(["data", "sexo", "idade"]).size()
obito_agrupado = obito_agrupado.to_frame(name = "obito")
obito_agrupado = obito_agrupado.reset_index().set_index("data")
top20 = bio["causa"].value_counts().head(20)
top20 = top20.index.tolist()
print(f"\n{green}top20:\n{reset}{top20}\n")
biotop20 = bio[bio["causa"].isin(top20)]
print(f"\n{green}biotop20:\n{reset}{biotop20}\n")
obito_agrupado_top20 = biotop20.groupby(["data", "sexo", "idade"]).size()
obito_agrupado_top20 = obito_agrupado_top20.to_frame(name = "obito")
obito_agrupado_top20 = obito_agrupado_top20.reset_index().set_index("data")
troca_sexo = {"Feminino" : "F", "Masculino" : "M"}
for sexo, letra in troca_sexo.items():
	obito_agrupado_top20 = obito_agrupado_top20.replace(sexo, letra)
obito_agrupado_top20["paciente"] = obito_agrupado_top20[["sexo", "idade"]].astype(str).agg("".join, axis = 1)
obito_agrupado_top20.drop(columns = ["sexo", "idade"], inplace = True)
obito_agrupado_top20 = obito_agrupado_top20.pivot_table(index = "data", columns = ["paciente"],
														values = "obito", aggfunc = "sum", fill_value = 0)
# Percentil 75
p75 = pd.read_csv(f"{caminho_indices}{p75}", low_memory = False)
print(f"\n{green}meteoro:\n{reset}{meteoro}\n")
print(f"\n{green}meteoro.info():\n{reset}{meteoro.info()}\n")
print(f"\n{green}obito_total:\n{reset}{obito_total}\n")
print(f"\n{green}p75:\n{reset}{p75}\n")
print(f"\n{green}obito_agrupado:\n{reset}{obito_agrupado}\n")
print(f"\n{green}obito_agrupado_top20:\n{reset}{obito_agrupado_top20}\n")
#sys.exit()

#Implementar com sexo/idade/diferença²/anomaliaestacionária.

### Montando Datasets
x1 = obito_total[["data","obitos"]]
x1["data"] = pd.to_datetime(x1["data"])
x2 = p75[["data", "total"]]
x2.rename(columns = {"total": "totalp75"}, inplace = True)
x2["data"] = pd.to_datetime(x2["data"])
x3 = p75[["data","I219"]]
x3.rename(columns = {"I219": "infarto_agudo_miocardio"}, inplace = True)
x3["data"] = pd.to_datetime(x3["data"])
print(f"\n{green}x1:\n{reset}{x1}\n")
print(f"\n{green}x1.info():\n{reset}{x1.info()}\n")
print(f"\n{green}x2:\n{reset}{x2}\n")
print(f"\n{green}x2.info():\n{reset}{x2.info()}\n")
print(f"\n{green}x3:\n{reset}{x3}\n")
print(f"\n{green}x3.info():\n{reset}{x3.info()}\n")
#dataset1 = x1.copy()
dataset1 = meteoro.merge(x1[["data", "obitos"]], how = "right", on = "data")
dataset1.dropna(inplace = True)
dataset_original1 = dataset1.copy()
#dataset2 = x2.copy()
dataset2 = meteoro.merge(x2[["data", "totalp75"]], how = "right", on = "data")
dataset2.dropna(inplace = True)
dataset_original2 = dataset2.copy()
#dataset3 = x3.copy()
dataset3 = meteoro.merge(x3[["data", "infarto_agudo_miocardio"]], how = "right", on = "data")
dataset3.dropna(inplace = True)
dataset_original3 = dataset3.copy()
print(f"\n{green}dataset1:\n{reset}{dataset1}\n")
print(f"\n{green}dataset1.info():\n{reset}{dataset1.info()}\n")
print(f"\n{green}dataset2:\n{reset}{dataset2}\n")
print(f"\n{green}dataset2.info():\n{reset}{dataset2.info()}\n")
print(f"\n{green}dataset3:\n{reset}{dataset3}\n")
print(f"\n{green}dataset3.info():\n{reset}{dataset3.info()}\n")
colunas_retroagir = meteoro.drop(columns = "data")
colunas_retroagir = colunas_retroagir.columns
print(f"\n{green}colunas_retroagir:\n{reset}{colunas_retroagir}\n")
#sys.exit()
"""
dataset1.to_csv(f"{caminho_dados}{_cidade}_dataset1.csv", index = False)
dataset2.to_csv(f"{caminho_dados}{_cidade}_dataset2.csv", index = False)
dataset3.to_csv(f"{caminho_dados}{_cidade}_dataset3.csv", index = False)

for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
    dataset[f"tmin_r{r}"] = dataset["tmin"].shift(-r)
    dataset[f"temp_r{r}"] = dataset["temp"].shift(-r)
    dataset[f"tmax_r{r}"] = dataset["tmax"].shift(-r)
    dataset[f"amplitude_t_r{r}"] = dataset["amplitude_t"].shift(-r)
    dataset[f"urmin_r{r}"] = dataset["urmin"].shift(-r)
    dataset[f"umidade_r{r}"] = dataset["umidade"].shift(-r)
    dataset[f"urmax_r{r}"] = dataset["urmax"].shift(-r)
    dataset[f"prec_r{r}"] = dataset["prec"].shift(-r)
    dataset[f"pressao_r{r}"] = dataset["pressao"].shift(-r)
    dataset[f"ventodir_r{r}"] = dataset["ventodir"].shift(-r)
    dataset[f"ventovel_r{r}"] = dataset["ventovel"].shift(-r)
#dataset.drop(columns = ["tmin", "temp", "tmax",
#						"urmin", "umidade", "urmax",
#						"prec", "pressao", "ventodir",  "ventovel"], inplace = True)
dataset.drop(columns = ["s_obito", "s_temp", "s_tmin", "s_tmax", "s_amplitude_t"], inplace = True)
dataset.dropna(inplace = True)
data_dataset = dataset.copy()
dataset.set_index("data", inplace = True)
dataset.columns.name = f"{cidade}"
"""

#sys.exit()
"""
### Dividindo Dataset em Treino e Teste

SEED = np.random.seed(0)
x = dataset.drop(columns = "obito")
y = dataset["obito"]
x_array = x.to_numpy().astype(int)
y_array = y.to_numpy().astype(int)
x_array = x_array.reshape(x_array.shape[0], -1)

treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array,
                                                        random_state = SEED,
                                                        test_size = 0.2)
"""
"""
z = 6850
x_ate_limite = x.iloc[:-z]
y_ate_limite = y.iloc[:-z]
xlimite = x.iloc[-z:]
ylimite = y.iloc[-z:]
treino_x = x_ate_limite.copy()
teste_x = xlimite.copy()
treino_y = y_ate_limite.copy()
teste_y = ylimite.copy()
"""
"""
sobre = RandomOverSampler(sampling_strategy = "minority")
sub = RandomUnderSampler(sampling_strategy = "majority")
treino_x, treino_y = sobre.fit_resample(treino_x, treino_y)
treino_x, treino_y = sub.fit_resample(treino_x, treino_y)

explicativas = x.columns.tolist() # feature_names = explicativas
treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
treino_x_explicado = treino_x_explicado.to_numpy().astype(int)
"""
#print(f"""Conjunto de Treino com as Variáveis Explicativas (<{limite}):\n{treino_x}\n
#Conjunto de Treino com as Variáveis Explicativas (>{fim}):\n{teste_x}\n 
#Conjunto de Teste com a Variável Dependente (<{limite}):\n{treino_y}\n 
#Conjunto de Teste com a Variável Dependente (>{fim}):\n{teste_y}\n
#Conjunto de Treino com as Variáveis Explicativas (Explicitamente Indicadas)(<{limite}):\n{treino_x_explicado}\n""")
#sys.exit()

"""
### Normalizando/Escalonando Dataset_x (Se Necessário)
escalonador = StandardScaler()
escalonador.fit(treino_x)
treino_normal_x = escalonador.transform(treino_x)
teste_normal_x = escalonador.transform(teste_x)

### Exibindo Informações
print("\n \n CONJUNTO DE DADOS PARA TREINO E TESTE \n")
print(dataset.info())
print("~"*80)
#print(dataset.dtypes)
#print("~"*80)
print(dataset)
#print("="*80)
#print(f"X no formato numpy.ndarray: {x_array}.")
print("="*80)
print(f"Treinando com {len(treino_x)} elementos e testando com {len(teste_x)} elementos.") # Tamanho é igual para dados normalizados
print(f"Formato dos dados (X) nas divisões treino: {treino_x.shape} e teste: {teste_x.shape}.")
print(f"Formato dos dados (Y) nas divisões treino: {treino_y.shape} e teste: {teste_y.shape}.")
print("="*80)

### Dividindo Dataset em Treino e Teste
SEED = np.random.seed(0)
x = dataset.drop(columns = "obito")
y = dataset["obito"]
x_array = x.to_numpy().astype(int)
y_array = y.to_numpy().astype(int)
x_array = x_array.reshape(x_array.shape[0], -1)

treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array,
                                                        random_state = SEED,
                                                        test_size = 0.2)
explicativas = x.columns.tolist() # feature_names = explicativas
treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
treino_x_explicado = treino_x_explicado.to_numpy().astype(int)

### Normalizando/Escalonando Dataset_x (Se Necessário)
escalonador = StandardScaler()
escalonador.fit(treino_x)
treino_normal_x = escalonador.transform(treino_x)
teste_normal_x = escalonador.transform(teste_x)
"""
"""
### Exibindo Informações
print("\n \n CONJUNTO DE DADOS PARA TREINO E TESTE \n")
print(dataset.info())
print("~"*80)
#print(dataset.dtypes)
#print("~"*80)
print(dataset)
print("="*80)
print("\ntreino_x\n", treino_x)
print("\nteste_x\n", teste_x)
print("\ntreino_y\n", treino_y)
print("\nteste_y\n" , teste_y)
#print(f"X no formato numpy.ndarray: {x_array}.")
print("="*80)
print(f"Treinando com {len(treino_x)} elementos e testando com {len(teste_x)} elementos.") # Tamanho é igual para dados normalizados
print(f"Formato dos dados (X) nas divisões treino: {treino_x.shape} e teste: {teste_x.shape}.")
print(f"Formato dos dados (Y) nas divisões treino: {treino_y.shape} e teste: {teste_y.shape}.")
print("="*80)
"""
#sys.exit()

#########################################################FUNÇÕES###############################################################
### Definições
def monta_dataset(dataset):
	dataset_montado = dataset.copy()
	for c in colunas_retroagir:
		for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
			dataset_montado[f"{c}_r{r}"] = dataset_montado[f"{c}"].shift(-r)
	dataset_montado.drop(columns = colunas_retroagir, inplace = True)
	dataset_montado.dropna(axis = 0, inplace = True)
	dataset_montado.set_index("data", inplace = True)
	dataset_montado.columns.name = f"{_CIDADE}"
	print(f"\n{green}dataset_montado:\n{reset}{dataset_montado}\n")
	print(f"\n{green}dataset_montado.info():\n{reset}{dataset_montado.info()}\n")
	return dataset_montado

def seleciona_periodo(dataset, str_periodo):
	dataset_periodo = dataset.copy()
	dataset_periodo.reset_index(inplace = True)
	dataset_periodo["data"] = pd.to_datetime(dataset_periodo["data"], errors = "coerce")
	dataset_periodo.set_index("data", inplace = True)
	print(f"\n{red}PERÍODO {green}{str_periodo.upper()} {red}SELECIONADO:\n{reset}{dataset_periodo}\n{dataset_periodo.info()}\n")
	if str_periodo == "quente":
		dataset_periodo = dataset_periodo[(dataset_periodo.index.month >= 10) & (dataset_periodo.index.month <= 3)]
	elif str_periodo == "frio":
		dataset_periodo = dataset_periodo[(dataset_periodo.index.month >= 5) & (dataset_periodo.index.month <= 9)]
	#dataset_periodo.reset_index(inplace = True)
	#dataset_periodo.drop(columns = "dia", inplace = True)
	#dataset_periodo.set_index("data", inplace = True)
	print(f"\n{green}PERÍODO {red}{str_periodo.upper()} {green}SELECIONADO:\n{reset}{dataset_periodo}\n{dataset_periodo.info()}\n")
	return dataset_periodo	

def treino_teste(n_dataset, dataset, cidade, tamanho_teste = 0.2):
	SEED = np.random.seed(0)
	if n_dataset == 1:
		x = dataset.drop(columns = "obitos")
		y = dataset["obitos"]
	elif n_dataset == 2:
		x = dataset.drop(columns = "totalp75")
		y = dataset["totalp75"]
	elif n_dataset == 3:
		x = dataset.drop(columns = "infarto_agudo_miocardio")
		y = dataset["infarto_agudo_miocardio"]
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	#x_array = x.to_numpy()
	#x_array = x_array.reshape(x_array.shape[0], -1)
	x_array = x.to_numpy().astype(int)
	y_array = y.to_numpy().astype(float)
	#x_array = x_array.reshape(x_array.shape[0], -1)
	treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array, # x_array, y_array, x, y, 
		                                                random_state = SEED,
		                                                test_size = tamanho_teste)
	explicativas = x.columns.tolist()
	treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
	treino_x_explicado = treino_x_explicado.to_numpy().astype(int)
	print(f"\n{green}explicativas:\n{reset}{explicativas}\n")
	print(f"\n{green}treino_x_explicado:\n{reset}{treino_x_explicado}\n")
	return x_array, y_array, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, explicativas, SEED

def escalona(treino_x, teste_x):
	escalonador = StandardScaler()
	escalonador.fit(treino_x)
	treino_normal_x = escalonador.transform(treino_x)
	teste_normal_x = escalonador.transform(teste_x)
	return treino_normal_x, teste_normal_x

def RF_modela_treina_preve(x, treino_x_explicado, treino_y, teste_x, SEED):
	modelo = RandomForestRegressor(n_estimators = 100, random_state = SEED)
	modelo.fit(treino_x_explicado, treino_y)
	y_previsto = modelo.predict(teste_x)
	previsoes = modelo.predict(x)
	previsoes = [int(p) for p in previsoes]
	print(f"\n{green}previsoes:\n{reset}{previsoes}\n")
	return modelo, y_previsto, previsoes

def RF_previsao_metricas(n_dataset, dataset, previsoes, n, teste_y, y_previsto):
	nome_modelo = "Random Forest"
	print("="*80)
	print(f"\n{nome_modelo.upper()} - {cidade}\n")
	if n_dataset == 1:
		lista_op = [f"obitos: {dataset['obitos'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
	elif n_dataset == 2:
		lista_op = [f"totalp75: {dataset['totalp75'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
	elif n_dataset == 3:
		lista_op = [f"Infarto Agudo: {dataset['infarto_agudo_miocardio'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	print("\n".join(lista_op))
	print("~"*80)
	EQM = mean_squared_error(teste_y, y_previsto)
	RQ_EQM = np.sqrt(EQM)
	R_2 = r2_score(teste_y, y_previsto).round(2)
	print(f"""
			\n MÉTRICAS {nome_modelo.upper()} - {cidade}
			\n Erro Quadrático Médio: {EQM}
			\n Coeficiente de Determinação (R²): {R_2}
			\n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM}
			""")
	print("="*80)
	return EQM, RQ_EQM, R_2

def lista_previsao(previsao, n, string_modelo):
	if string_modelo not in ["RF", "NN"]:
		print("!!"*80)
		print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
		print("!!"*80)
		sys.exit()
	nome_modelo = "Random Forest" if string_modelo == "RF" else "Rede Neural"
	previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
	print("="*80)
	print(f"\n{nome_modelo.upper()} - {cidade}\n")
	lista_op = [f"Óbitos: {dataset['obito'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
	print("\n".join(lista_op))
	print("="*80)

def grafico_previsao(n_dataset, dataset, previsao, R_2):
	# Gráfico de Comparação entre Observação e Previsão dos Modelos
	nome_modelo = "Random Forest"
	final = pd.DataFrame()
	dataset.reset_index(inplace = True)
	final["Data"] = dataset["data"]
	if n_dataset == 1:
		final["obito"] = dataset["obitos"]
		nome_arquivo = "totais"
	elif n_dataset == 2:
		final["obito"] = dataset["totalp75"]
		nome_arquivo = "totalp75"
	elif n_dataset == 3:
		final["obito"] = dataset["infarto_agudo_miocardio"]
		nome_arquivo = "I219"
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	final.reset_index(inplace = True)
	final.drop(columns = "index", inplace = True)
	print(final)
	#sys.exit()
	final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
	final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
	print(final)
	previsoes = previsao
	previsoes = previsoes[:len(final)]
	final["Previstos"] = previsoes
	final["Data"] = pd.to_datetime(final["Data"])
	final.replace([np.inf, -np.inf], np.nan, inplace=True)
	print(final)
	print("="*80)
	plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
	sns.lineplot(x = final["Data"], y = final["obito"], # linestyle = "--" linestyle = "-."
		     	color = "darkblue", linewidth = 1, label = "Observado")
	sns.lineplot(x = final["Data"], y = final["Previstos"],
		     	color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
	plt.title(f"MODELO {nome_modelo.upper()} (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
	plt.xlabel("Série Histórica (Observação Diária)")
	plt.ylabel(f"Número de Óbitos Cardiovasculares ({nome_arquivo})")
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
	   'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
	 	'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
	 	'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
	 	'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U', 
	 	'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	_cidade = cidade
	for velho, novo in troca.items():
		_cidade = _cidade.replace(velho, novo)
	nome_arquivo = f"modelo_RF_{nome_arquivo}_{_cidade}.pdf"
	if _SALVAR == True:
		plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
		print(f"{green}\nSALVANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
	if _VISUALIZAR == True:	
		print(f"{green}\nVISUALIZANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
		plt.show()

def metricas(string_modelo, modeloNN = None):
	if string_modelo not in ["RF", "NN"]:
		print("!!"*80)
		print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
		print("!!"*80)
		sys.exit()
	elif string_modelo == "NN":
		if modeloNN is None:
			print("!!"*80)
			raise ValueError("'modeloNN' não foi fornecido para a função metricas() do modelo de rede neural!")
		else:
			sumario = []
			modeloNN.summary(print_fn = lambda x: sumario.append(x))
			sumario = "\n".join(sumario)
			print(f"\n MÉTRICAS REDE NEURAL\n \n {sumario}")
	else:
		print(f"""
				\n MÉTRICAS RANDOM FOREST - {cidade}
				\n Erro Quadrático Médio: {EQM_RF}
				\n Coeficiente de Determinação (R²): {R_2}
				\n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM_RF}
				""")

def metricas_importancias(n_dataset, modeloRF, explicativas, teste_x, teste_y):
	importancias = modeloRF.feature_importances_
	importancias = importancias.round(4)
	indices = np.argsort(importancias)[::-1]
	variaveis_importantes = pd.DataFrame({"Variáveis": explicativas, "Importâncias": importancias})
	variaveis_importantes = variaveis_importantes.sort_values(by = "Importâncias", ascending = False)
	importancia_impureza = pd.Series(importancias, index = explicativas)
	print(variaveis_importantes)
	#1 Impurezas
	std = np.std([tree.feature_importances_ for tree in modeloRF.estimators_], axis=0)
	fig, ax = plt.subplots(figsize = (10, 6), layout = "tight", frameon = False)
	importancia_impureza = importancia_impureza.sort_values(ascending = False)
	importancia_impureza[:10].plot.bar(yerr = std[:10], ax = ax)
	if n_dataset == 1:
		nome_arquivo = "totais"
	elif n_dataset == 2:
		nome_arquivo = "totalp75"
	elif n_dataset == 3:
		nome_arquivo = "I219"
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	ax.set_title(f"VARIÁVEIS IMPORTANTES PARA MODELO RANDOM FOREST\nMUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel("Perda de Impureza Média")
	ax.set_xlabel(f"Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares ({nome_arquivo})")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 60)
	for i, v in enumerate(importancia_impureza[:10].values):
		ax.text(i + 0.01, v, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	nome_arquivo = f"importancias_modelo_RF_{nome_arquivo}_{_cidade}.pdf"
	if _SALVAR == True:
		plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
		print(f"{green}\nSALVANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
	if _VISUALIZAR == True:
		print(f"{green}\nVISUALIZANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
		plt.show()
	#2 Permutações
	n_permuta = 10
	resultado_permuta = permutation_importance(modeloRF, teste_x, teste_y, n_repeats = n_permuta, random_state = SEED, n_jobs = 2)
	importancia_permuta = pd.Series(resultado_permuta.importances_mean, index = explicativas)
	importancia_permuta = importancia_permuta.sort_values(ascending = False)
	std = resultado_permuta.importances_std
	fig, ax = plt.subplots(figsize = (10, 6), layout = "tight", frameon = False)
	importancia_permuta[:10].plot.bar(yerr = std[:10], ax = ax)
	if n_dataset == 1:
		nome_arquivo = f"totais"
	elif n_dataset == 2:
		nome_arquivo = f"totalp75"
	elif n_dataset == 3:
		nome_arquivo = f"I219"
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	ax.set_title(f"VARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO ({n_permuta})\nMUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel("Acurácia Média")
	ax.set_xlabel(f"Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares ({nome_arquivo})")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 60)
	for i, v in enumerate(importancia_permuta[:10].values):
		ax.text(i + 0.01, v, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	nome_arquivo = f"importancias_permuta_modelo_RF_{nome_arquivo}_{_cidade}.pdf"
	if _SALVAR == True:
		plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
		print(f"{green}\nSALVANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
	if _VISUALIZAR == True:
		print(f"{green}\nVISUALIZANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
		plt.show()
	print(f"\n{green}VARIÁVEIS IMPORTANTES:\n{reset}{importancia_impureza}\n")
	print(f"\n{green}VARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO:\n{reset}{importancia_permuta}")
	return importancias, indices, variaveis_importantes

def metrica_shap(n_dataset, modelo, treino_x, teste_x):
	expl_shap = shap.Explainer(modelo, treino_x)
	valor_shap = expl_shap(teste_x)
	plt.figure(figsize = (10, 6)).set_layout_engine(None)#, layout = "constrained", frameon = False).set_layout_engine(None)
	if n_dataset == 1:
		nome_arquivo = "totais"
	elif n_dataset == 2:
		nome_arquivo = "totalp75"
	elif n_dataset == 3:
		nome_arquivo = "I219"
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	ax = plt.gca()
	ax.set_title(f"SHAP (SHapley Additive exPlanations) PARA MODELO RANDOM FOREST\nMUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel("Valor SHAP")
	ax.set_xlabel(f"Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares ({nome_arquivo})")
	ax.set_facecolor("honeydew")
	plt.rcParams.update({"figure.autolayout" : False})
	shap.summary_plot(valor_shap, teste_x)#, legacy_colorbar = True)
	nome_arquivo = f"importancias_SHAP_modelo_RF_{nome_arquivo}_{_cidade}.pdf"
	if _SALVAR == True:
		plt.savefig(f'{caminho_shap}{nome_arquivo}', format = "pdf", dpi = 1200)
		print(f"{green}\nSALVANDO:\n{caminho_shap}{nome_arquivo}{reset}")
	if _VISUALIZAR == True:
		print(f"{green}\nVISUALIZANDO:\n{caminho_shap}{nome_arquivo}{reset}")
		plt.show()
	print(f"\n{green}VARIÁVEIS IMPORTANTES E VALORES (SHAP):\n{reset}{valor_shap}\n")

def caminho_decisao(x, modelo, explicativas):
		#amostra = x.iloc[0].values.reshape(1, -1)
		#caminho, _ = modelo.decision_path(amostra)
		#caminho_denso = caminho.toarray()
		unica_arvore = modelo.estimators_[0]
		relatorio_decisao = export_text(unica_arvore, feature_names = explicativas,
										spacing = 5, decimals = 0, show_weights = True)
		plt.figure(figsize = (25, 10), layout = "constrained", frameon = False)
		ax = plt.gca()
		for i, child in enumerate(ax.get_children()):
			if isinstance(child, plt.Line2D):
				if i % 2 == 0:
					child.set_color("red")
				else:
					child.set_color("blue")
		plt.title(f"ÁRVORE DE DECISÃO DO MODELO RANDOM FOREST.\nMUNICÍPIO DE {cidade}, RIO GRANDE DO SUL.")
		plot_tree(unica_arvore, feature_names = explicativas, filled = True, rounded = True, fontsize = 6,
					proportion = True, node_ids = True, precision = 0, impurity = False)#, max_depth = 6) # impureza = ErroQuadrático
		ax.set_facecolor("honeydew")
		if _SALVAR == True:
			plt.savefig(f'{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.pdf', format = "pdf", dpi = 1200)
			print(f"\n{ansi['green']}ARQUIVO SALVO COM SUCESSO\n\n{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.pdf{ansi['reset']}\n")
			with open(f'{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.txt', 'w') as file:
				file.write(relatorio_decisao)
			print(f"\n{ansi['green']}ARQUIVO SALVO COM SUCESSO\n\n{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.txt{ansi['reset']}\n")
		if _VISUALIZAR == True:
			print("\n\n{ansi['green']}RELATÓRIO DA ÁRVORE DE DECISÃO\n\n{cidade}\n\n{cidade}{ansi['reset']}\n\n", relatorio_decisao)
			plt.show()
		#print("\n\nCAMINHO DE DECISÃO\n\n", caminho_denso)
		return unica_arvore, relatorio_decisao #amostra, caminho, caminho_denso

def salva_modeloRF(n_dataset, modelo, cidade):
	if not os.path.exists(caminho_modelos):
		os.makedirs(caminho_modelos)
	if n_dataset == 1:
		nome_arquivo = f"RF_obitos_r{_RETROAGIR}_{_cidade}.h5"
	elif n_dataset == 2:
		nome_arquivo = f"RF_totalp75_r{_RETROAGIR}_{_cidade}.h5"
	elif n_dataset == 3:
		nome_arquivo = f"RF_I219_r{_RETROAGIR}_{_cidade}.h5"
	else:
		print(f"\n{red}DATASET NÃO ENCONTRADO!\n{reset}")
	joblib.dump(modelo, f"{caminho_modelos}{nome_arquivo}")
	print(f"\n{green}MODELO RANDOM FOREST DE {cidade} SALVO!\n{reset}")
	print(f"\n{green}Caminho e Nome:\n{bold} {caminho_modelos}{nome_arquivo}\n{reset}")
	print("\n" + f"{green}={cyan}={reset}"*80 + "\n")

######################################################RANDOM_FOREST############################################################
"""
### Instanciando e Treinando Modelo Regressor Random Forest
modeloRF = RandomForestRegressor(n_estimators = 100, random_state = SEED) #n_estimators = número de árvores
modeloRF.fit(treino_x_explicado, treino_y)
#modeloRF.salva_modelo("RF")
joblib.dump(modeloRF, f"{caminho_modelos}RF_obitos_r{_RETROAGIR}_{_cidade}.h5")

### Testando e Avaliando
y_previstoRF = modeloRF.predict(teste_x)
EQM_RF = mean_squared_error(teste_y, y_previstoRF)
RQ_EQM_RF = np.sqrt(EQM_RF)
R_2 = round(r2_score(teste_y, y_previstoRF), 2)#.round(2) 

### Testando e Validando Modelo
testesRF = modeloRF.predict(teste_x)
previsoesRF = modeloRF.predict(x)
previsoesRF = [int(p) for p in previsoesRF]

### Exibindo Informações, Gráficos e Métricas
lista_previsao(previsoesRF, 5, "RF")
grafico_previsao(previsoesRF, testesRF, "RF")
metricas("RF")

importancias, indices, variaveis_importantes =  metricas_importancias(modeloRF, explicativas)
unica_arvore, relatorio_decisao = caminho_decisao(x, modeloRF, explicativas)
sys.exit()
"""
#########################################################AUTOMATIZANDO###############################################################
if _AUTOMATIZAR == True:
	for cidade in cidades:
		#caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indice_amplitude/"
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/anomalia_estacionaria/"
		if not os.path.exists(caminho_resultados):
			os.makedirs(caminho_resultados)
		#caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/indice_amplitude/"
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/anomalia_estacionaria/"
		if not os.path.exists(caminho_modelos):
			os.makedirs(caminho_modelos)		
		dataset_inicio1 = dataset_original1.copy()
		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()
		dataset1 = monta_dataset(dataset_inicio1)
		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)
		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset1, cidade)
		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset3, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset1, previsoes1, 5, teste_y1, y_previsto1)
		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset3, previsoes3, 5, teste_y3, y_previsto3)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)
		caminho_shap = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/SHAP/"
		if not os.path.exists(caminho_shap):
			os.makedirs(caminho_shap)
		metrica_shap(1, modelo1, treino_x_explicado1, teste_x1)
		pd.DataFrame(explicativas1)
		explicativas1.to_csv(f"{caminho_shap}explicativas1.csv")#, index = False)
		metrica_shap(2, modelo2, treino_x_explicado2, teste_x2)
		pd.DataFrame(explicativas2)
		explicativas2.to_csv(f"{caminho_shap}explicativas2.csv")#, index = False)
		metrica_shap(3, modelo3, treino_x_explicado3, teste_x3)
		pd.DataFrame(explicativas3)
		explicativas3.to_csv(f"{caminho_shap}explicativas3.csv")#, index = False)
		"""
		grafico_previsao(1, dataset1, previsoes1, R_2_1)
		grafico_previsao(2, dataset2, previsoes2, R_2_2)
		grafico_previsao(3, dataset3, previsoes3, R_2_3)
		"""
		#sys.exit()
		salva_modeloRF(1, modelo1, cidade)
		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)
		"""
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/periodo_quente/"
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/periodo_quente/"
		print(f"\n{green}dataset_original1:\n{reset}{dataset_original1}\n")

		dataset_inicio1 = dataset_original1.copy()
		dataset1 = monta_dataset(dataset_inicio1)

		dataset_q1 = seleciona_periodo(dataset1, "quente")

		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset_q1, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset_q1, previsoes1, 5, teste_y1, y_previsto1)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		grafico_previsao(1, dataset_q1, previsoes1, R_2_1)
		salva_modeloRF(1, modelo1, cidade)

		print(f"\n{green}dataset_original2:\n{reset}{dataset_original2}\n")
		print(f"\n{green}dataset_original3:\n{reset}{dataset_original3}\n")

		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()

		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)

		dataset_q2 = seleciona_periodo(dataset2, "quente")
		dataset_q3 = seleciona_periodo(dataset3, "quente")

		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset_q2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset_q3, cidade)

		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)

		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset_q2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset_q3, previsoes3, 5, teste_y3, y_previsto3)

		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)

		grafico_previsao(2, dataset_q2, previsoes2, R_2_2)
		grafico_previsao(3, dataset_q3, previsoes3, R_2_3)
		#sys.exit()

		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)
		"""
		#caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indice_amplitude_pfrio/"
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/anomalia_estacionaria_pfrio/"
		if not os.path.exists(caminho_resultados):
			os.makedirs(caminho_resultados)
		#caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/indice_amplitude_pfrio/"
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/anomalia_estacionaria_pfrio/"
		if not os.path.exists(caminho_modelos):
			os.makedirs(caminho_modelos)
		dataset_inicio1 = dataset_original1.copy()
		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()
		dataset1 = monta_dataset(dataset_inicio1)
		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)
		dataset_f1 = seleciona_periodo(dataset1, "frio")
		dataset_f2 = seleciona_periodo(dataset2, "frio")
		dataset_f3 = seleciona_periodo(dataset3, "frio")
		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset_f1, cidade)
		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset_f2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset_f3, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset_f1, previsoes1, 5, teste_y1, y_previsto1)
		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset_f2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset_f3, previsoes3, 5, teste_y3, y_previsto3)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)
		metrica_shap(1, modelo1, treino_x1, teste_x1)
		metrica_shap(2, modelo2, treino_x2, teste_x2)
		metrica_shap(3, modelo3, treino_x3, teste_x3)
		"""
		grafico_previsao(1, dataset_f1, previsoes1, R_2_1)
		grafico_previsao(2, dataset_f2, previsoes2, R_2_2)
		grafico_previsao(3, dataset_f3, previsoes3, R_2_3)
		"""
		#sys.exit()
		salva_modeloRF(1, modelo1, cidade)
		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)

##### ANOMALIA
print(f"\n{green}meteoro_origem\n{reset}{meteoro_origem}\n")
print(f"\n{green}anomalia\n{reset}{anomalia}\n")
meteoro = meteoro_origem.merge(anomalia, left_on = "dia", how = "left", suffixes = ("", "_anomal"), right_index = True)
meteoro.drop(columns = ["dia", "dia_anomal", "Unnamed: 0"], inplace = True) #"Unnamed: 0_anomal"
print(f"\n{green}meteoro\n{reset}{meteoro}\n")
#sys.exit()
#
# Saúde
print(f"\n{green}bio:\n{reset}{bio}\n")
# Percentil 75
print(f"\n{green}meteoro:\n{reset}{meteoro}\n")
print(f"\n{green}meteoro.info():\n{reset}{meteoro.info()}\n")
print(f"\n{green}obito_total:\n{reset}{obito_total}\n")
print(f"\n{green}p75:\n{reset}{p75}\n")
print(f"\n{green}obito_agrupado:\n{reset}{obito_agrupado}\n")
print(f"\n{green}obito_agrupado_top20:\n{reset}{obito_agrupado_top20}\n")
#sys.exit()

#Implementar com sexo/idade/diferença²/anomaliaestacionária.

### Montando Datasets
x1 = obito_total[["data","obitos"]]
x1["data"] = pd.to_datetime(x1["data"])
x2 = p75[["data", "total"]]
x2.rename(columns = {"total": "totalp75"}, inplace = True)
x2["data"] = pd.to_datetime(x2["data"])
x3 = p75[["data","I219"]]
x3.rename(columns = {"I219": "infarto_agudo_miocardio"}, inplace = True)
x3["data"] = pd.to_datetime(x3["data"])
print(f"\n{green}x1:\n{reset}{x1}\n")
print(f"\n{green}x1.info():\n{reset}{x1.info()}\n")
print(f"\n{green}x2:\n{reset}{x2}\n")
print(f"\n{green}x2.info():\n{reset}{x2.info()}\n")
print(f"\n{green}x3:\n{reset}{x3}\n")
print(f"\n{green}x3.info():\n{reset}{x3.info()}\n")
#dataset1 = x1.copy()
dataset1 = meteoro.merge(x1[["data", "obitos"]], how = "right", on = "data")
dataset1.dropna(inplace = True)
dataset_original1 = dataset1.copy()
#dataset2 = x2.copy()
dataset2 = meteoro.merge(x2[["data", "totalp75"]], how = "right", on = "data")
dataset2.dropna(inplace = True)
dataset_original2 = dataset2.copy()
#dataset3 = x3.copy()
dataset3 = meteoro.merge(x3[["data", "infarto_agudo_miocardio"]], how = "right", on = "data")
dataset3.dropna(inplace = True)
dataset_original3 = dataset3.copy()
print(f"\n{green}dataset1:\n{reset}{dataset1}\n")
print(f"\n{green}dataset1.info():\n{reset}{dataset1.info()}\n")
print(f"\n{green}dataset2:\n{reset}{dataset2}\n")
print(f"\n{green}dataset2.info():\n{reset}{dataset2.info()}\n")
print(f"\n{green}dataset3:\n{reset}{dataset3}\n")
print(f"\n{green}dataset3.info():\n{reset}{dataset3.info()}\n")
colunas_retroagir = meteoro.drop(columns = "data")
colunas_retroagir = colunas_retroagir.columns
print(f"\n{green}colunas_retroagir:\n{reset}{colunas_retroagir}\n")
#sys.exit()


if _AUTOMATIZAR == True:
	for cidade in cidades:
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/anomalia/"
		if not os.path.exists(caminho_resultados):
			os.makedirs(caminho_resultados)
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/anomalia/"
		if not os.path.exists(caminho_modelos):
			os.makedirs(caminho_modelos)		
		dataset_inicio1 = dataset_original1.copy()
		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()
		dataset1 = monta_dataset(dataset_inicio1)
		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)
		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset1, cidade)
		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset3, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset1, previsoes1, 5, teste_y1, y_previsto1)
		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset3, previsoes3, 5, teste_y3, y_previsto3)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)
		"""
		grafico_previsao(1, dataset1, previsoes1, R_2_1)
		grafico_previsao(2, dataset2, previsoes2, R_2_2)
		grafico_previsao(3, dataset3, previsoes3, R_2_3)
		"""
		#sys.exit()
		salva_modeloRF(1, modelo1, cidade)
		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)
		"""
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/periodo_quente/"
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/periodo_quente/"
		print(f"\n{green}dataset_original1:\n{reset}{dataset_original1}\n")

		dataset_inicio1 = dataset_original1.copy()
		dataset1 = monta_dataset(dataset_inicio1)

		dataset_q1 = seleciona_periodo(dataset1, "quente")

		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset_q1, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset_q1, previsoes1, 5, teste_y1, y_previsto1)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		grafico_previsao(1, dataset_q1, previsoes1, R_2_1)
		salva_modeloRF(1, modelo1, cidade)

		print(f"\n{green}dataset_original2:\n{reset}{dataset_original2}\n")
		print(f"\n{green}dataset_original3:\n{reset}{dataset_original3}\n")

		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()

		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)

		dataset_q2 = seleciona_periodo(dataset2, "quente")
		dataset_q3 = seleciona_periodo(dataset3, "quente")

		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset_q2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset_q3, cidade)

		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)

		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset_q2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset_q3, previsoes3, 5, teste_y3, y_previsto3)

		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)

		grafico_previsao(2, dataset_q2, previsoes2, R_2_2)
		grafico_previsao(3, dataset_q3, previsoes3, R_2_3)
		#sys.exit()

		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)
		"""
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/anomalia_pfrio/"
		if not os.path.exists(caminho_resultados):
			os.makedirs(caminho_resultados)
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/anomalia_pfrio/"
		if not os.path.exists(caminho_modelos):
			os.makedirs(caminho_modelos)
		dataset_inicio1 = dataset_original1.copy()
		dataset_inicio2 = dataset_original2.copy()
		dataset_inicio3 = dataset_original3.copy()
		dataset1 = monta_dataset(dataset_inicio1)
		dataset2 = monta_dataset(dataset_inicio2)
		dataset3 = monta_dataset(dataset_inicio3)
		dataset_f1 = seleciona_periodo(dataset1, "frio")
		dataset_f2 = seleciona_periodo(dataset2, "frio")
		dataset_f3 = seleciona_periodo(dataset3, "frio")
		x1, y1, treino_x1, teste_x1, treino_y1, teste_y1, treino_x_explicado1, explicativas1, SEED = treino_teste(1, dataset_f1, cidade)
		x2, y2, treino_x2, teste_x2, treino_y2, teste_y2, treino_x_explicado2, explicativas2, SEED = treino_teste(2, dataset_f2, cidade)
		x3, y3, treino_x3, teste_x3, treino_y3, teste_y3, treino_x_explicado3, explicativas3, SEED = treino_teste(3, dataset_f3, cidade)
		modelo1, y_previsto1, previsoes1 = RF_modela_treina_preve(x1, treino_x_explicado1, treino_y1, teste_x1, SEED)
		modelo2, y_previsto2, previsoes2 = RF_modela_treina_preve(x2, treino_x_explicado2, treino_y2, teste_x2, SEED)
		modelo3, y_previsto3, previsoes3 = RF_modela_treina_preve(x3, treino_x_explicado3, treino_y3, teste_x3, SEED)
		EQM1, RQ_EQM1, R_2_1 = RF_previsao_metricas(1, dataset_f1, previsoes1, 5, teste_y1, y_previsto1)
		EQM2, RQ_EQM2, R_2_2 = RF_previsao_metricas(2, dataset_f2, previsoes2, 5, teste_y2, y_previsto2)		
		EQM3, RQ_EQM3, R_2_3 = RF_previsao_metricas(3, dataset_f3, previsoes3, 5, teste_y3, y_previsto3)
		metricas_importancias(1, modelo1, explicativas1, teste_x1, teste_y1)
		metricas_importancias(2, modelo2, explicativas2, teste_x2, teste_y2)
		metricas_importancias(3, modelo3, explicativas3, teste_x3, teste_y3)
		"""
		grafico_previsao(1, dataset_f1, previsoes1, R_2_1)
		grafico_previsao(2, dataset_f2, previsoes2, R_2_2)
		grafico_previsao(3, dataset_f3, previsoes3, R_2_3)
		"""
		#sys.exit()
		salva_modeloRF(1, modelo1, cidade)
		salva_modeloRF(2, modelo2, cidade)
		salva_modeloRF(3, modelo3, cidade)
		
######################################################################################################################################

