### Bibliotecas Correlatas
import matplotlib.pyplot as plt 
import matplotlib as mpl             
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels as sm
import pymannkendall as mk
#from scipy.stats import linregress
import esmtools.stats as esm
import xarray as xr
### Suporte
import sys
import os

### CONDIÇÕES PARA VARIAR ##################################
                                                       #####
_LOCAL = "SIFAPSC" # OPÇÕES >>> GH|CASA|CLUSTER|SIFAPSC#####
                                                       #####
##################### Valores Booleanos #######        ##### # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False  ####        #####
                                           ####        #####
_VISUALIZAR = sys.argv[2]    # True|False  ####        #####
                                           ####        #####
_SALVAR = sys.argv[3]        # True|False  ####        #####
###############################################        #####
                                                       #####
############################################################

_LIMIAR_RETRO = True
_CLIMA = True
_ENTOMOEPIDEMIO = True
_LIMIAR_TMIN = True
_LIMIAR_TMAX = True
_LIMIAR_PREC = True

_RETROAGIR = 16 # Semanas Epidemiológicas
_ANO = "2023" # "2023" # "2022" # "2021" # "2020" # "total"
_CIDADE = "Florianópolis" #"Florianópolis"#"Itajaí"#"Joinville"#"Chapecó"
_METODO = "spearman" # "pearson" # "spearman" # "kendall"

_CIDADE = _CIDADE.upper()
##### Padrão ANSI ##################################
ansi = {"bold" : "\033[1m", "red" : "\033[91m",
        "green" : "\033[92m", "yellow" : "\033[33m",
        "blue" : "\033[34m", "magenta" : "\033[35m",
        "cyan" : "\033[36m", "white" : "\033[37m", "reset" : "\033[0m"}
#################################################################################

match _LOCAL:
	case "GH":
		caminho_dados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/dados/"
		caminho_resultados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/resultados/porto_alegre/"
		caminho_modelos = "https://github.com/matheusf30/RS_saude_precisao/tree/main/modelos/"
	case "SIFAPSC":
		caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
		caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
		caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"
	case "CLUSTER":
		caminho_dados = "..."
	case "CASA":
		caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/"
		caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/modelos/"
	case _:
		print("CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteo_poa_h_96-22.csv"
bio = "obito_cardiovasc_total_poa_96-22.csv"

### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", skiprows = 10, sep = ";", low_memory = False)
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)

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

# Tratando Sazonalidade
timeindex = biometeoro.copy()
timeindex = timeindex.set_index("data")
timeindex["dia"] = timeindex.index.dayofyear
#timeindex["mes_dia"] = timeindex.index.to_period('M').astype(str) + '-' + timeindex.index.day.astype(str)
print(timeindex, timeindex.info())
#sys.exit()
print("="*80)
media_dia = timeindex.groupby("dia").mean().round(2)
media_dia.reset_index(inplace = True)
print(media_dia)
print(media_dia.index)
#media_dia[["obito","tmin","tmax"]].plot()
#plt.show()
componente_sazonal = timeindex.merge(media_dia, left_on = "dia", how = "left", suffixes = ("", "_media"), right_index = True)
sem_sazonal = pd.DataFrame(index = timeindex.index)
print(componente_sazonal)
for coluna in timeindex.columns:
	if coluna in componente_sazonal.columns:
		media_coluna = f"{coluna}_media"
		if media_coluna in componente_sazonal.columns:
			sem_sazonal[coluna] = timeindex[coluna] - componente_sazonal[media_coluna]
		else:
			print(f"{ansi['red']}Coluna {media_coluna} não encontrada no componente sazonal!{ansi['reset']}")
	else:
		print(f"{ansi['red']}Coluna {coluna} não encontrada no timeindex!{ansi['reset']}")
sem_sazonal.drop(columns = "dia", inplace = True)
print(sem_sazonal)
print(sem_sazonal.columns)
#sem_sazonal[["obito","tmin","tmax"]].plot()
#plt.show()

# Verificando Tendência
colunas = ['obito', 'tmin', 'temp', 'tmax', 'amplitude_t',
			'urmin', 'umidade', 'urmax', 'prec',
			'pressao', 'ventodir', 'ventovel']
for c in colunas:
	tendencia = mk.original_test(sem_sazonal[c])
	if tendencia.trend == "decreasing":
		print(f"\n{ansi['red']}{c}\n{tendencia.trend}{ansi['reset']}\n")
	if tendencia.trend == "no trend":
		print(f"\n{ansi['cyan']}{c}\n{tendencia.trend}{ansi['reset']}\n")
	elif tendencia.trend == "increasing":
		print(f"\n{ansi['green']}{c}\n{tendencia.trend}{ansi['reset']}\n")
#	else:
#		print(f"\n{ansi['magenta']}NÃO EXECUTANDO\n{c}{ansi['reset']}\n")

# Tratando Tendência
# anomalia_estacionaria = dados - ( a + b * x )
anomalia_estacionaria = pd.DataFrame()
for c in colunas:
	print(c)
	print(sem_sazonal[c])
	tendencia = mk.original_test(sem_sazonal[c])
	print(tendencia)
	sem_tendencia = sem_sazonal[c] -(tendencia.slope + tendencia.intercept)# * len(sem_sazonal[c]))
	anomalia_estacionaria[c] = sem_tendencia
print(f"{ansi['green']}\nsem_sazonal\n{ansi['reset']}", sem_sazonal)
print(f"{ansi['green']}\nanomalia_estacionaria\n{ansi['reset']}", anomalia_estacionaria)

"""
sys.exit()

# Tratando Tendência
print("="*80, "\nTratando Tendência\n")
print("\nsem_sazonal.shape\n", sem_sazonal.shape)
print("\nlen(sem_sazonal.index)\n", len(sem_sazonal.index))
print("\nlen(sem_sazonal.columns)\n", len(sem_sazonal.columns))
print("\nsem_sazonal.index\n", sem_sazonal.index)
print("\nsem_sazonal.columns\n", sem_sazonal.columns)
array = xr.DataArray(sem_sazonal.values,
					dims = ["data", "variable"],
					coords = {"data": sem_sazonal.index,
							"variable": sem_sazonal.columns})
print("\nArray\n", array)
print("\narray.variable.values\n", array.variable.values)
sys.exit()
resultados = {"slope": {},
			"intercept": {},
			"r_value": {},
			"p_value": {},
			"std_err": {} }
for var in array.variable.values:
	try:
		y = array.sel(variable=var).values
		x = np.arange(len(array.data))
		slope, intercept, r_value, p_value, std_err = linregress(x, y)
		resultados["slope"][var] = slope
		resultados["intercept"][var] = intercept
		resultados["r_value"][var] = r_value
		resultados["p_value"][var] = p_value
		resultados["std_err"][var] = std_err
	except KeyError:
		print(f"Variable '{var}' not found in the DataArray.")
	except Exception as e:
		print(f"An error occurred for variable '{var}': {e}")
df_resultados = pd.DataFrame(resultados)
print("\nResultados\n", df_resultados)

sys.exit()
#tendencia = esm.linregress(x = array, y = array, dim = "data")

print(tendencia)
print("="*80)
sys.exit()

print("="*80)
componente_sazonal = pd.DataFrame(index = timeindex.index)
# Extract the day of the year for each date
componente_sazonal['dia'] = timeindex.index.dayofyear
# Merge with media_dia to get daily mean values
componente_sazonal = componente_sazonal.join(media_dia, on = 'dia', how = 'left', rsuffix = '_mean')
# Calculate the seasonality by subtracting daily means from the original data
print(componente_sazonal, componente_sazonal.info())
sys.exit()
for col in timeindex.columns:
	if col in componente_sazonal.columns:
		componente_sazonal[col] = timeindex[col] - media_dia[col]
print(componente_sazonal)
"""
#################################################################################
### Correlações
#################################################################################

colunas_r = ['tmin', 'temp', 'tmax', 'amplitude_t',
			'urmin', 'umidade', 'urmax', 'prec',
			'pressao', 'ventodir', 'ventovel']
_retroceder = 3

for _r in range(1, _retroceder +1):
	for c_r in colunas_r:
		anomalia_estacionaria[f"{c_r}_r{_r}"] = anomalia_estacionaria[f"{c_r}"].shift(-_r)
print(anomalia_estacionaria)
correlacao_dataset = anomalia_estacionaria.corr()
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
fig.suptitle(f"MATRIZ DE CORRELAÇÃO", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
plt.show()

biometeoro_set = biometeoro.copy()
for _r in range(1, _retroceder +1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
correlacao_dataset = biometeoro_set.corr()
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
fig.suptitle(f"MATRIZ DE CORRELAÇÃO", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
plt.show()

sys.exit()








#################################################################################
### Correlacionando (Focos, Casos E Limiares, Retroações)
#################################################################################
# VARIÁVES, LIMIARES E RETROAÇÕES CORRELACIONADOS
if _AUTOMATIZA == True and _LIMIAR_RETRO == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	limiares_tmax = [22, 24, 26, 28, 30, 32]
	limiares_tmin = [14, 16, 18, 20, 22, 24]
	limiares_prec = [5, 20, 35, 50, 65, 80, 95]
	lista_retro = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for r in lista_retro:
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "CASOS"}#,  f"{_CIDADE}" : f"L{_LIMIAR}_PREC"}
				dataset.rename(columns = troca_nome, inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				print(dataset)
				### Incluindo limiares
				for _LIMIAR in limiares_tmin:
					print(_LIMIAR)
					limite = tmin_sem.copy()
					limite.set_index("Data", inplace = True)
					limite.drop(columns = "tmin", inplace = True)
					print(f"{ansi['red']}\nLIMIAR TMIN > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
					limite.reset_index(inplace = True)
					print(f"{ansi['yellow']}\nLIMIAR TMIN > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite["Data"] = pd.to_datetime(limite["Data"])
					limite = limite.sort_values(by = ["Data"])
					limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
					limite = limite.groupby(["Semana"]).sum(numeric_only = True)
					limite.reset_index(inplace = True)
					limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
					dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
					dataset.rename(columns = {f"{_CIDADE}" : f"L{_LIMIAR}_TMIN"}, inplace = True)
					dataset[f"L{_LIMIAR}_TMIN_r{r}"] = dataset[f"L{_LIMIAR}_TMIN"].shift(-r)
					dataset.drop(columns = f"L{_LIMIAR}_TMIN", inplace = True)
					print(f"{ansi['green']}\nLIMIAR TMIN > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
				for _LIMIAR in limiares_tmax:
					print(_LIMIAR)
					limite = tmax_sem.copy()
					limite.set_index("Data", inplace = True)
					limite.drop(columns = "tmax", inplace = True)
					print(f"{ansi['red']}\nLIMIAR TMAX > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
					limite.reset_index(inplace = True)
					print(f"{ansi['yellow']}\nLIMIAR TMAX > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite["Data"] = pd.to_datetime(limite["Data"])
					limite = limite.sort_values(by = ["Data"])
					limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
					limite = limite.groupby(["Semana"]).sum(numeric_only = True)
					limite.reset_index(inplace = True)
					limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
					dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
					dataset.rename(columns = {f"{_CIDADE}" : f"L{_LIMIAR}_TMAX"}, inplace = True)
					dataset[f"L{_LIMIAR}_TMAX_r{r}"] = dataset[f"L{_LIMIAR}_TMAX"].shift(-r)
					dataset.drop(columns = f"L{_LIMIAR}_TMAX", inplace = True)
					print(f"{ansi['green']}\nLIMIAR TMAX > {_LIMIAR} C\n{limite}\n{ansi['reset']}")
					print(limite.info())
				for _LIMIAR in limiares_prec:
					print(_LIMIAR)
					limite = prec_sem.copy()
					limite.set_index("Data", inplace = True)
					limite.drop(columns = "prec", inplace = True)
					limite.dropna(inplace = True)
					print(f"{ansi['red']}\nLIMIAR PREC > {_LIMIAR} mm\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite.dropna(inplace = True)
					limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
					limite.reset_index(inplace = True)
					print(f"{ansi['yellow']}\nLIMIAR PREC > {_LIMIAR} mm\n{limite}\n{ansi['reset']}")
					print(limite.info())
					limite["Data"] = pd.to_datetime(limite["Data"])
					limite = limite.sort_values(by = ["Data"])
					limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
					limite = limite.groupby(["Semana"]).sum(numeric_only = True)
					limite.reset_index(inplace = True)
					limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
					limite.drop([0], axis = 0, inplace = True)
					dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
					dataset.rename(columns = {f"{_CIDADE}" : f"L{_LIMIAR}_PREC"}, inplace = True)
					dataset[f"L{_LIMIAR}_PREC_r{r}"] = dataset[f"L{_LIMIAR}_PREC"].shift(-r)
					dataset.drop(columns = f"L{_LIMIAR}_PREC", inplace = True)
					print(f"{ansi['green']}\nLIMIAR PREC > {_LIMIAR} mm\n{limite}\n{ansi['reset']}")
					print(limite.info())
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				print(dataset)
				print(dataset.info())
				#sys.exit()
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
					print("\n\n2023\n\n")
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
					print("\n\n2022\n\n")
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
					print("\n\n2021\n\n")
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
					print("\n\n2020\n\n")
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.drop(columns = "Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
                            vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					if r == 0:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; sem retroagir)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
					elif r == 1:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {r} semana epidemiológica)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
					else:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {r} semanas epidemiológicas)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
				else:
					if r == 0:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; sem retroagir)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
					elif r == 1:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; retroagindo {r} semana epidemiológica)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
					else:
						fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e LIMIARES CLIMATOLÓGICOS** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; retroagindo {r} semanas epidemiológicas)\n**(Temperatura Mínima (C), Temperatura Máxima (C) e Precipitação(mm))", weight = "bold", size = "medium")
					_cidade = _CIDADE
					troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
							'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
							'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
							'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
							'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
							'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				if _SALVAR == True:
					for velho, novo in troca.items():
						_cidade = _cidade.replace(velho, novo)
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/retrolimiar/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_retrolimiar_{_cidade}_{_ANO}_r{r}s.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_retrolimiar_{_cidade}_{_ANO}_r{r}s.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_retrolimiar_{_cidade}_{_ANO}_r{r}s.pdf{ansi['reset']}\n")
					plt.show()

#################################################################################
### Correlacionando (VARIÁVEIS CLIMATOLÓGICAS)
#################################################################################
if _AUTOMATIZA == True and _CLIMA == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			#Automatizando essas listas acima...
			### Montando dataset
			dataset = tmin[["Semana"]].copy()
			dataset["TMIN"] = tmin[_CIDADE].copy()
			dataset["TMED"] = tmed[_CIDADE].copy()
			dataset["TMAX"] = tmax[_CIDADE].copy()
			dataset = dataset.merge(prec[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
			dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
			dataset.dropna(axis = 0, inplace = True)
			dataset = dataset.iloc[:, :].copy()
			dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
			troca_nome = {f"{_CIDADE}_x" : "PREC", f"{_CIDADE}_y" : "FOCOS", f"{_CIDADE}" : "CASOS"}
			dataset = dataset.rename(columns = troca_nome)
			dataset.fillna(0, inplace = True)
			if _ANO == "2023":
				dataset = dataset.iloc[-53:, :].copy()
			elif _ANO == "2022":
				dataset = dataset.iloc[-105:-53, :].copy()
			elif _ANO == "2021":
				dataset = dataset.iloc[-157:-105, :].copy()
			elif _ANO == "2020":
				dataset = dataset.iloc[-209:-157, :].copy()
			else:
				print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
			dataset.dropna(inplace = True)
			dataset.set_index("Semana", inplace = True)
			dataset.columns.name = f"{_CIDADE}"
			ordem_colunas = ["FOCOS", "CASOS", "TMIN", "TMED", "TMAX", "PREC"]
			dataset = dataset.reindex(columns = ordem_colunas)
			print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
			print(dataset.info())
			print("~"*80)
			print(dataset.dtypes)
			print("~"*80)
			print(dataset)
			#sys.exit()
			### Retroagindo dataset
			for r in range(1, _RETROAGIR + 1):
				#dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
				dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
				#dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
				dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
				#dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
				#dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
			dataset.dropna(inplace = True)
			dataset.columns.name = f"{_CIDADE}"
			### Matriz de Correlações
			correlacao_dataset = dataset.corr(method = f"{_METODO}")
			print("="*80)
			print(f"Método de {_METODO.title()} \n", correlacao_dataset)
			print("="*80)
			
			# Gerando Visualização (.pdf) da Matriz
			fig, ax = plt.subplots(figsize = (16, 8), layout = "constrained", frameon = False)
			filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
			sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
			ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
			ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
			if _ANO == "total":
				fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS E VARIÁVEIS CLIMATOLÓGICAS EM {_CIDADE} \n *(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas)", weight = "bold", size = "medium")
			else:
				fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS E VARIÁVEIS CLIMATOLÓGICAS EM {_CIDADE} \n *(Método de {_METODO.title()}; durante {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas)", weight = "bold", size = "medium")
			if _SALVAR == True:
				_cidade = _CIDADE
				troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
						'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
						'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
						'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
						'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
						'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				for velho, novo in troca.items():
					_cidade = _cidade.replace(velho, novo)
				caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/clima/"
				os.makedirs(caminho_correlacao, exist_ok = True)
				plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
				print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf{ansi['reset']}\n""")
			if _VISUALIZAR == True:
				print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf{ansi['reset']}\n")
				plt.show()

#################################################################################
### Correlacionando (VARIÁVEIS ENTOMO-EPIDEMIOLÓGICAS)
#################################################################################
if _AUTOMATIZA == True and _ENTOMOEPIDEMIO == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			#Automatizando essas listas acima...
			### Montando dataset
			dataset = tmin[["Semana"]].copy()
			dataset["TMIN"] = tmin[_CIDADE].copy()
			dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
			dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
			dataset.dropna(axis = 0, inplace = True)
			dataset = dataset.iloc[:, :].copy()
			troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "CASOS"}
			dataset = dataset.rename(columns = troca_nome)
			dataset.fillna(0, inplace = True)
			if _ANO == "2023":
				dataset = dataset.iloc[-53:, :].copy()
			elif _ANO == "2022":
				dataset = dataset.iloc[-105:-53, :].copy()
			elif _ANO == "2021":
				dataset = dataset.iloc[-157:-105, :].copy()
			elif _ANO == "2020":
				dataset = dataset.iloc[-209:-157, :].copy()
			else:
				print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
			dataset.dropna(inplace = True)
			dataset.set_index("Semana", inplace = True)
			dataset.columns.name = f"{_CIDADE}"
			dataset.drop(columns = "TMIN", inplace = True)
			print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
			print(dataset.info())
			print("~"*80)
			print(dataset.dtypes)
			print("~"*80)
			### Retroagindo dataset
			#_RETROAGIR = 20
			for r in range(1, _RETROAGIR + 1):
				dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
				dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
			dataset.dropna(inplace = True)
			dataset.columns.name = f"{_CIDADE}"
			### Matriz de Correlações
			correlacao_dataset = dataset.corr(method = f"{_METODO}")
			print("="*80)
			print(f"Método de {_METODO.title()} \n", correlacao_dataset)
			print("="*80)
			#print(dataset)
			#sys.exit()			
			# Gerando Visualização (.pdf) da Matriz
			fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
			filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
			sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
			ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
			ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
			if _ANO == "total":
				fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS e CASOS (VARIÁVEIS ENTOMO-EPIDEMIOLÓGICAS) EM {_CIDADE} \n *(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas)", weight = "bold", size = "medium")
			else:
				fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS e CASOS (VARIÁVEIS ENTOMO-EPIDEMIOLÓGICAS) EM {_CIDADE} \n *(Método de {_METODO.title()}; durante {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas)", weight = "bold", size = "medium")
			if _SALVAR == True:
				_cidade = _CIDADE
				troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
						'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
						'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
						'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
						'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
						'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				for velho, novo in troca.items():
					_cidade = _cidade.replace(velho, novo)
				caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/entomoepidemio/"
				os.makedirs(caminho_correlacao, exist_ok = True)
				plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_fococaso_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
				print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
NOME DO ARQUIVO: matriz_correlacao_{_METODO}_fococaso_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf{ansi['reset']}\n""")
			if _VISUALIZAR == True:
				print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_fococaso_{_cidade}_r{_RETROAGIR}s_{_ANO}.pdf{ansi['reset']}\n")
				plt.show()

#################################################################################
### Correlacionando (Índice Clima e Focos)
#################################################################################
if _AUTOMATIZA == True and _iCLIMA == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	lista_constante = [1, 2, 3, 4, 5, 6, 7, 8]
	_K = 1
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for _K in lista_constante:
				print(_K)
				#Automatizando essas listas acima...
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset["iCLIMA"] =  np.cbrt((tmin[_CIDADE].rolling(_K).mean() ** _K) * (prec[_CIDADE].rolling(_K).mean() / _K))
				dataset = dataset.merge(prec[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "PREC"}
				dataset = dataset.rename(columns = troca_nome)
				dataset.fillna(0, inplace = True)
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				dataset.drop(columns = "PREC", inplace = True)
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				#sys.exit()
				### Retroagindo dataset
				#_RETROAGIR = 20
				for r in range(1, _RETROAGIR + 1):
					dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
					dataset[f"iCLIMA_r{r}"] = dataset["iCLIMA"].shift(-r)
				dataset.dropna(inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS e Índice Climatológico** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas; k = {_K})\n**np.cbrt((tmin.rolling(k).mean() ** k) * (prec.rolling(k).mean() / k))", weight = "bold", size = "medium")
				else:
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS e Índice Climatológico** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas; k = {_K})\n**np.cbrt((tmin.rolling(k).mean() ** k) * (prec.rolling(k).mean() / k))", weight = "bold", size = "medium")
				if _SALVAR == True:
					_cidade = _CIDADE
					troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
							'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
							'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
							'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
							'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
							'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
					for velho, novo in troca.items():
						_cidade = _cidade.replace(velho, novo)
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/indiceclimato/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_iclima_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_iclima_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_iclima_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf{ansi['reset']}\n")
					plt.show()

#################################################################################
### Correlacionando (Índice Epidemio e Casos)
#################################################################################
if _AUTOMATIZA == True and _iEPIDEMIO == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	lista_constante = [1, 2, 3, 4, 5, 6, 7, 8]
	_K = 1
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for _K in lista_constante:
				print(_K)
				#Automatizando essas listas acima...
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				troca_nome = {f"{_CIDADE}_x" : "CASOS", f"{_CIDADE}_y" : "FOCOS"}
				dataset = dataset.rename(columns = troca_nome)
				dataset["iEPIDEMIO"] =  np.sqrt((dataset["FOCOS"].rolling(_K).mean() / _K) * dataset["CASOS"].rolling(_K).mean())
				"""
				print(dataset)
				print(dataset.info())
				sys.exit()
				#dataset["iCLIMA"] =  (tmin[_CIDADE].rolling(_K).mean() ** _K) * (prec[_CIDADE].rolling(_K).mean() / _K)
				"""
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				dataset.fillna(0, inplace = True)
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				#sys.exit()
				### Retroagindo dataset
				#_RETROAGIR = 20
				for r in range(1, _RETROAGIR + 1):
					dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
					dataset[f"iEPIDEMIO_r{r}"] = dataset["iEPIDEMIO"].shift(-r)
				dataset.dropna(inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre CASOS e Índice Epidemiológico** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas; k = {_K})\n**(a definir)", weight = "bold", size = "medium")
				else:
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre CASOS e Índice Epidemiológico** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas; k = {_K})\n**(a definir)", weight = "bold", size = "medium")
				if _SALVAR == True:
					_cidade = _CIDADE
					troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
							'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
							'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
							'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
							'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
							'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
					for velho, novo in troca.items():
						_cidade = _cidade.replace(velho, novo)
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/indiceepidemio/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_iepidemio_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_iepidemio_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_iepidemio_{_cidade}_r{_RETROAGIR}s_{_ANO}_k{_K}.pdf{ansi['reset']}\n")
					plt.show()

#################################################################################
### Correlacionando (Focos, Casos E Limiares TMIN)
#################################################################################
if _AUTOMATIZA == True and _LIMIAR_TMIN == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	limiares = [14, 16, 18, 20, 22, 24]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for _LIMIAR in limiares:
				print(_LIMIAR)
				limite = tmin_sem.copy()
				limite.set_index("Data", inplace = True)
				limite.drop(columns = "tmin", inplace = True)
				limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
				limite.reset_index(inplace = True)
				limite["Data"] = pd.to_datetime(limite["Data"])
				limite = limite.sort_values(by = ["Data"])
				limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
				limite = limite.groupby(["Semana"]).sum(numeric_only = True)
				limite.reset_index(inplace = True)
				limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
				limite.drop([0], axis = 0, inplace = True)
				print(limite.info())
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				print(dataset.info())
				dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "CASOS",  f"{_CIDADE}" : f"L{_LIMIAR}_TMIN"}
				dataset = dataset.rename(columns = troca_nome)
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				dataset.fillna(0, inplace = True)
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				print(dataset)
				#sys.exit()
				### Retroagindo dataset
				#_RETROAGIR = 20
				for r in range(1, _RETROAGIR + 1):
					dataset[f"L{_LIMIAR}_TMIN_r{r}"] = dataset[f"L{_LIMIAR}_TMIN"].shift(-r)
				dataset.dropna(inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Temperatura Mínima** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Temperatura Mínima > {_LIMIAR} C)\n", weight = "bold", size = "medium")
				else:
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Temperatura Mínima** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Temperatura Mínima > {_LIMIAR} C)", weight = "bold", size = "medium")
					_cidade = _CIDADE
					troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
							'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
							'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
							'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
							'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
							'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				if _SALVAR == True:
					for velho, novo in troca.items():
						_cidade = _cidade.replace(velho, novo)
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/limiares_tmin/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_tmin_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_tmin_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_tmin_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n")
					plt.show()

#################################################################################
### Correlacionando (Focos, Casos E Limiares TMAX)
#################################################################################
if _AUTOMATIZA == True and _LIMIAR_TMAX == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	limiares = [22, 24, 26, 28, 30, 32]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for _LIMIAR in limiares:
				print(_LIMIAR)
				limite = tmax_sem.copy()
				limite.set_index("Data", inplace = True)
				limite.drop(columns = "tmax", inplace = True)
				limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
				limite.reset_index(inplace = True)
				limite["Data"] = pd.to_datetime(limite["Data"])
				limite = limite.sort_values(by = ["Data"])
				limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
				limite = limite.groupby(["Semana"]).sum(numeric_only = True)
				limite.reset_index(inplace = True)
				limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
				limite.drop([0], axis = 0, inplace = True)
				print(limite.info())
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "CASOS",  f"{_CIDADE}" : f"L{_LIMIAR}_TMAX"}
				dataset = dataset.rename(columns = troca_nome)
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				dataset.fillna(0, inplace = True)
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				print(dataset)
				#sys.exit()
				### Retroagindo dataset
				#_RETROAGIR = 20
				for r in range(1, _RETROAGIR + 1):
					dataset[f"L{_LIMIAR}_TMAX_r{r}"] = dataset[f"L{_LIMIAR}_TMAX"].shift(-r)
				dataset.dropna(inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Temperatura Máxima** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Temperatura Máxima < {_LIMIAR} C)\n", weight = "bold", size = "medium")
				else:
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Temperatura Máxima** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Temperatura Máxima < {_LIMIAR} C)", weight = "bold", size = "medium")
				_cidade = _CIDADE
				troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
						'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
						'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
						'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
						'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
						'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				for velho, novo in troca.items():
					_cidade = _cidade.replace(velho, novo)
				if _SALVAR == True:
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/limiares_tmax/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_tmax_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_tmax_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_tmax_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n")
					plt.show()

#################################################################################
### Correlacionando (Focos, Casos E Limiares PREC)
#################################################################################
if _AUTOMATIZA == True and _LIMIAR_PREC == True:
	lista_cidades = ["Florianópolis", "Itajaí", "Joinville", "Chapecó"]
	lista_anos = ["2023", "2022", "2021", "2020", "total"]
	limiares = [5, 10, 15, 20, 25, 30, 35]
	for _CIDADE in lista_cidades:
		_CIDADE = _CIDADE.upper()
		print(_CIDADE)
		for _ANO in lista_anos:
			print(_ANO)
			for _LIMIAR in limiares:
				print(_LIMIAR)
				limite = prec_sem.copy()
				limite.set_index("Data", inplace = True)
				limite.drop(columns = "prec", inplace = True)
				limite = limite.applymap(lambda x: 1 if x > _LIMIAR else 0)
				limite.reset_index(inplace = True)
				limite["Data"] = pd.to_datetime(limite["Data"])
				limite = limite.sort_values(by = ["Data"])
				limite["Semana"] = limite["Data"].dt.to_period("W-SAT").dt.to_timestamp()
				limite = limite.groupby(["Semana"]).sum(numeric_only = True)
				limite.reset_index(inplace = True)
				limite["Semana"] = limite["Semana"].dt.strftime("%Y-%m-%d")
				limite.drop([0], axis = 0, inplace = True)
				print(limite.info())
				### Montando dataset
				dataset = tmin[["Semana"]].copy()
				dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset = dataset.merge(limite[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
				dataset.dropna(axis = 0, inplace = True)
				troca_nome = {f"{_CIDADE}_x" : "FOCOS", f"{_CIDADE}_y" : "CASOS",  f"{_CIDADE}" : f"L{_LIMIAR}_PREC"}
				dataset = dataset.rename(columns = troca_nome)
				dataset.dropna(axis = 0, inplace = True)
				dataset = dataset.iloc[:, :].copy()
				dataset.fillna(0, inplace = True)
				if _ANO == "2023":
					dataset = dataset.iloc[-53:, :].copy()
				elif _ANO == "2022":
					dataset = dataset.iloc[-105:-53, :].copy()
				elif _ANO == "2021":
					dataset = dataset.iloc[-157:-105, :].copy()
				elif _ANO == "2020":
					dataset = dataset.iloc[-209:-157, :].copy()
				else:
					print(f"{ansi['red']}{_ANO} fora da abordagem desse roteiro!\n\n{ansi['cyan']}Por favor, recodifique-o ou utilize um dos seguintes anos:\n{ansi['green']}\n2020\n2021\n2022\n2023\n\nA correlação será realizada pela SÉRIE HISTÓRICA {ansi['magenta']} intencionalmente!{ansi['reset']}")
				dataset.dropna(inplace = True)
				dataset.set_index("Semana", inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				print(f"\n \n DATASET PARA INICIAR MATRIZ DE CORRELAÇÃO ({_METODO.title()}) \n")
				print(dataset.info())
				print("~"*80)
				print(dataset.dtypes)
				print("~"*80)
				print(dataset)
				#sys.exit()
				### Retroagindo dataset
				#_RETROAGIR = 20
				for r in range(1, _RETROAGIR + 1):
					dataset[f"L{_LIMIAR}_PREC_r{r}"] = dataset[f"L{_LIMIAR}_PREC"].shift(-r)
				dataset.dropna(inplace = True)
				dataset.columns.name = f"{_CIDADE}"
				### Matriz de Correlações
				correlacao_dataset = dataset.corr(method = f"{_METODO}")
				print("="*80)
				print(f"Método de {_METODO.title()} \n", correlacao_dataset)
				print("="*80)
				#print(dataset)
				#sys.exit()			
				# Gerando Visualização (.pdf) da Matriz
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _ANO == "total":
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Precipitação** EM {_CIDADE}\n*(Método de {_METODO.title()}; durante a série histórica; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Precipitação > {_LIMIAR} mm)\n", weight = "bold", size = "medium")
				else:
					fig.suptitle(f"MATRIZ DE CORRELAÇÃO* entre FOCOS, CASOS e Limiar de Precipitação** EM {_CIDADE}\n*(Método de {_METODO.title()}; em {_ANO}; retroagindo {_RETROAGIR} semanas epidemiológicas) **(Limiar de Precipitação > {_LIMIAR} mm)", weight = "bold", size = "medium")
					_cidade = _CIDADE
					troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
							'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
							'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
							'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
							'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
							'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
				if _SALVAR == True:
					for velho, novo in troca.items():
						_cidade = _cidade.replace(velho, novo)
					caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/limiares_prec/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_prec_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_prec_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n""")
				if _VISUALIZAR == True:
					print(f"{ansi['cyan']} Visualizando: matriz_correlacao_{_METODO}_prec_{_cidade}_r{_RETROAGIR}s_{_ANO}_LIMIAR{_LIMIAR}.pdf{ansi['reset']}\n")
					plt.show()
