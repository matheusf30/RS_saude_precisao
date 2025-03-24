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

### Encaminhamento aos Diretórios
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_indices = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
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
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
# BIO-SAÚDE
bio.rename(columns = {"CAUSABAS" : "causa"}, inplace = True)
bio["data"] = pd.to_datetime(bio[["anoobito", "mesobito", "diaobito"]].astype(str).agg("-".join, axis = 1), format = "%Y-%m-%d")
bio.reset_index(inplace = True)
bio["obito"] = np.ones(len(bio)).astype(int)
bio.drop(columns=["CODMUNRES", "diaobito", "mesobito", "anoobito"], inplace = True)
bio = bio[["data", "obito", "sexo", "idade", "causa"]].sort_values(by = "data")
#bio = bio.groupby(by = ["data"])["obito"].sum()
total = bio.groupby(by = ["data"])["obito"].sum()

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
print(f"\n{green}DADOS METEOROLÓGICOS:\n{reset}{meteoro}\n{meteoro.dtypes}\n")
print(f"\n{green}DADOS DE ÓBITOS:\n{reset}{total}\n{total.dtypes}\n")


# BIOMETEORO E SELEÇÃO DE PERÍODOS
meteoro.reset_index(inplace = True)
meteoro["data"] = pd.to_datetime(meteoro["data"])
total = total.to_frame(name = "obito")
total.reset_index(inplace = True)
biometeoro = meteoro.merge(total, on = "data", how = "inner")
biometeoro = biometeoro[["data", "obito", "heat_index", "wind_chill",
						"tmin", "temp", "tmax", "amplitude_t",
						"urmin", "umidade", "urmax",
						"prec", "pressao", "ventodir", "ventovel"]]
print(f"\n{green}DADOS DE ÓBITOS (BIOMETEOROLOGIA):\n{reset}{biometeoro}\n{biometeoro.dtypes}\n")
antes10 = biometeoro[biometeoro["data"].dt.year <= 2010]
print(f"\n{green}DADOS DE ÓBITOS (BIOMETEOROLOGIA < 2010):\n{reset}{antes10}\n{antes10.dtypes}\n")
depois10 = biometeoro[biometeoro["data"].dt.year >= 2011]
print(f"\n{green}DADOS DE ÓBITOS (BIOMETEOROLOGIA > 2011):\n{reset}{depois10}\n{depois10.dtypes}\n")

### Visualização Gráfica
# Antes de 2010
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = antes10["data"], y = antes10["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = antes10["data"], y = antes10["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = antes10["data"], y = antes10["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA ANTES DE 2010, {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_antes2010_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
# Depois de 2010
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = depois10["data"], y = depois10["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = depois10["data"], y = depois10["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = depois10["data"], y = depois10["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA DEPOIS DE 2010, {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_depois2010_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
# Série Total
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = biometeoro["data"], y = biometeoro["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = biometeoro["data"], y = biometeoro["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = biometeoro["data"], y = biometeoro["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA, {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_serietotal_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
################ MÉDIAS MÓVEIS #######################################
media_movel = 30
colunas = biometeoro.columns[1:]
antes10_mm = antes10[colunas].rolling(media_movel).mean()
depois10_mm = depois10[colunas].rolling(media_movel).mean()
biometeoro_mm = biometeoro[colunas].rolling(media_movel).mean()
# Antes de 2010
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = antes10["data"], y = antes10_mm["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = antes10["data"], y = antes10_mm["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = antes10["data"], y = antes10_mm["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA ANTES DE 2010 (MÉDIA MÓVEL), {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_antes2010_mediamovel_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
# Depois de 2010
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = depois10["data"], y = depois10_mm["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = depois10["data"], y = depois10_mm["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = depois10["data"], y = depois10_mm["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA DEPOIS DE 2010 (MÉDIA MÓVEL), {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_depois2010_mediamovel_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
# Série Total
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = biometeoro["data"], y = biometeoro_mm["tmin"], zorder = 1,
	        color = "blue", alpha = 0.7, linewidth = 1,
			label = "Temperatura Mínima", ax = ax1)
sns.lineplot(x = biometeoro["data"], y = biometeoro_mm["tmax"], zorder = 1,
	        color = "red", alpha = 0.7, linewidth = 1,
			label = "Temperatura Máxima", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = biometeoro["data"], y = biometeoro_mm["obito"], zorder = 2,
	         color = "black", alpha = 0.8, linewidth = 1, label = "Óbitos", ax = ax2)
ax2.set_title(f"ANÁLISE EXPLORATÓRIA DE TEMPERATURAS E ÓBITOS CARDIOVASCULARES.\nSÉRIE HISTÓRICA (MÉDIA MÓVEL), {cidade.upper()}, RIO GRANDE DO SUL.")
ax1.legend(loc = "upper left")
ax1.grid(True)
ax2.legend(loc = "upper right")
ax2.grid(True)
ax1.set_facecolor("honeydew")
ax2.set_facecolor("honeydew")
nome_arquivo = f"analise_exploratoria_serietotal_mediamovel_{_cidade}_.pdf"
if _VISUALIZAR == True:
	print(f"\n{green}VISUALIZANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}\n")
	plt.show()
if _SALVAR == True:
	plt.savefig(f'{caminho_resultados}{nome_arquivo}', format = "pdf", dpi = 1200)
	print(f"{green}SALVANDO:\n\n{caminho_resultados}{nome_arquivo}{reset}")
"""
# Visualização Prévia
_sigma = int(input(f"\n{cyan}>>> Caso a suavização não seja necessária, digite zero (0).\n>>> Caso seja, selecione um número inteiro maior que zero: {reset}"))
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
	if _VISUALIZAR == "True":
		print(f"{green}VISUALIZANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{reset}")
		plt.show()
	if _SALVAR == "True":
		plt.savefig(f'{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf', format = "pdf", dpi = 1200)
		print(f"{green}SALVANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{reset}")

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
	if _VISUALIZAR == "True":
		print(f"{green}VISUALIZANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{reset}")
		plt.show()
	if _SALVAR == "True":
		plt.savefig(f'{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf', format = "pdf", dpi = 1200)
		print(f"{green}SALVANDO:\n\n{caminho_resultados}analise_exploratoria_serie_historica_{_cidade}_.pdf{reset}")
"""


