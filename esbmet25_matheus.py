### Bibliotecas Correlatas
import matplotlib.pyplot as plt 
import matplotlib as mpl             
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import cm
import matplotlib.colors as cls
import matplotlib.dates as mdates  
import cmocean
from datetime import timedelta
import numpy as np
import seaborn as sns
import statsmodels as sm
import pymannkendall as mk
import xarray as xr
### Suporte
import sys
import os
### Tratando avisos
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


### Condições para Variar #######################################################

_LOCAL = "IFSC" # OPÇÕES>>> "GH" "CASA" "IFSC"
"""
##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################
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
if _LOCAL == "GH": # _ = Variável Privada
	caminho_dados = "https://raw.githubusercontent.com/matheusf30/dados_dengue/main/"
	caminho_modelos = "https://github.com/matheusf30/dados_dengue/tree/main/modelos"
elif _LOCAL == "IFSC":
	caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
	caminho_shp = "/home/sifapsc/scripts/matheus/dados_dengue/shapefiles/"
	caminho_modelos = "/home/sifapsc/scripts/matheus/dados_dengue/modelos/"
	caminho_resultados = "/home/sifapsc/scripts/matheus/dengue/resultados/modelagem/"
	caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/"
	caminho_cartografia = "/home/sifapsc/scripts/matheus/dengue/resultados/cartografia/"
else:
	print(f"\n{red}CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!{reset}")
print(f"\n{green}OS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n{reset}\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
biometeoro = "biometeoro_PORTO_ALEGRE.csv"
clima = "climatologia.csv"

### Abrindo Arquivo
biometeoro = pd.read_csv(f"{caminho_dados}{biometeoro}", low_memory = False)
clima = pd.read_csv(f"{caminho_dados}{clima}", low_memory = False)

print(f"\n{green}BIOMETEOROLOGIA\n{reset}{biometeoro}\n")
print(f"\n{green}SAZONALIDADE\n{reset}{clima}\n")

#sys.exit()

### Pré-Processamento (Série Histórica)
#biometeoro[["heat_index", "wind_chill"]] = (biometeoro[["heat_index", "wind_chill"]] - 32) * 5 / 9 # ºF >> ºC
biometeoro["data"] = pd.to_datetime(biometeoro["data"], errors = "coerce")
biometeoro = biometeoro.dropna(subset = ["data"])
print(biometeoro[pd.isna(biometeoro["data"])])
biometeoro.set_index("data", inplace = True)
print(f"\n{red}PERÍODO {green}NÃO {red}SELECIONADO:\n{reset}{biometeoro}\n{biometeoro.info()}\n")
biometeoro1 = biometeoro[(biometeoro.index.year <= 2011)]
biometeoro1.reset_index(inplace = True)
biometeoro2 = biometeoro[(biometeoro.index.year > 2011)]
biometeoro2.reset_index(inplace = True)
print(f"\n{green}PORTO ALEGRE (Série Temporal)\n{reset}{biometeoro}\n")
print(f"\n{green}PORTO ALEGRE (2001-2011)\n{reset}{biometeoro1}\n")
print(f"\n{green}PORTO ALEGRE (2012-2022)\n{reset}{biometeoro2}\n")
#print(f"\n{green}PORTO ALEGRE (Índices Simplificados)\n{reset}{biometeoro[['heat_index', 'wind_chill']]}\n")
#sys.exit()

periodos = [biometeoro1, biometeoro2]
for idx, biometeoro0 in enumerate(periodos):

	### Visualização Gráfica
	fig, axs = plt.subplots(2, 1, figsize = (12, 6), layout = "tight", frameon = False,  sharex = True)
	axs[0].set_facecolor("honeydew") #.gcf()
	axs[0].legend(loc = "upper center")
	axs[0].legend(loc = "lower left")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["tmax"].rolling(30).mean(),  ax = axs[0],
					color = "red", linewidth = 1.5, label = "Temperatura Máxima")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["temp"].rolling(30).mean(),  ax = axs[0],
					color = "orange", linewidth = 1.5, label = "Temperatura Média")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["tmin"].rolling(30).mean(),  ax = axs[0],
					color = "darkblue", linewidth = 1.5, label = "Temperatura Mínima")
	"""
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["heat_index"],  ax = axs[0],
					color = "red", linewidth = 1.8, linestyle = ":", label = "Heat Index")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["wind_chill"],  ax = axs[0],
					color = "darkblue", linewidth = 1.8, linestyle = ":", label = "Wind Chill")
	"""
	axs[0].set_ylabel("Temperaturas (C)")
	ax2 = axs[0].twinx()
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["obito"].rolling(30).mean(), ax = ax2,
					color = "purple", linewidth = 1, linestyle = "--", label = "Óbitos Cardiovasculares")
	ax2.fill_between(biometeoro0["data"], biometeoro0["obito"].rolling(30).mean(), color = "purple", alpha = 0.3)
	ax2.set_ylabel("Óbitos Cardiovasculares")
	ax2.legend(loc = "lower right")
	axs[1].set_facecolor("honeydew")
	ax3 = axs[1].twinx()
	ax3.set_ylabel("Precipitação (mm)")
	ax3.legend(loc = "upper right")
	"""
	sns.barplot(x = biometeoro0["data"], y = biometeoro0["prec"],  ax = ax3,
					color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
	
	sns.barplot(x = biometeoro0["data"].astype(str), ci = None, y = biometeoro0["prec"],  ax = ax3,
					color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
	"""
	ax3.bar(biometeoro0["data"], biometeoro0["prec"].rolling(30).mean(),#  ax = ax3,
					color = "royalblue", width = 1.5, alpha = 0.5, label = "Precipitação")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["urmax"].rolling(30).mean(),  ax = axs[1],
					color = "red", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Máxima")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["umidade"].rolling(30).mean(),  ax = axs[1],
					color = "orange", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa")
	sns.lineplot(x = biometeoro0["data"], y = biometeoro0["urmin"].rolling(30).mean(),  ax = axs[1],
					color = "darkblue", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Mínima")
	axs[1].set_ylabel("Umidade Relativa (%)")
	axs[1].set_ylim(50, 105)
	axs[1].legend(loc = "upper left")
	axs[1].grid(False)
	"""
	meses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
	nomes_meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
	plt.xticks(meses, nomes_meses)
	"""
	axs[1].set_xlabel("Série Temporal")
	#fig.suptitle(f"CASOS DE DENGUE, FOCOS DE _Aedes_ sp., TEMPERATURAS (MÍNIMA, MÉDIA E MÁXIMA) E PRECIPITAÇÃO.\nSAZONALIDADE POR MÉDIAS SEMANAIS PARA O MUNICÍPIO DE biometeoro0, SANTA CATARINA.")
	nome_arquivo = f"esbmet25_ciclo_anual_subplots_biometeoro0_{idx}.pdf"
	caminho_esbmet = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/esbmet/"
	#if _SALVAR == True:
	os.makedirs(caminho_esbmet, exist_ok = True)
	plt.savefig(f'{caminho_esbmet}{nome_arquivo}', format = "pdf", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!\n
	{cyan}ENCAMINHAMENTO: {caminho_esbmet}\n
	NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
	#if _VISUALIZAR == True:
	print(f"\n{cyan}Visualizando:\n{caminho_esbmet}{nome_arquivo}\n{reset}")
	plt.show()


### Pré-Processamento (Climatologia DIA)
clima[["heat_index", "wind_chill"]] = (clima[["heat_index", "wind_chill"]] - 32) * 5 / 9 # ºF >> ºC

### Visualização Gráfica (SAZONALIDADE)
fig, axs = plt.subplots(2, 1, figsize = (12, 6), layout = "tight", frameon = False,  sharex = True)
axs[0].set_facecolor("honeydew") #.gcf()
sns.lineplot(x = clima.index, y = clima["tmax"],  ax = axs[0],
				color = "red", linewidth = 1.5, label = "Temperatura Máxima")
sns.lineplot(x = clima.index, y = clima["temp"],  ax = axs[0],
				color = "orange", linewidth = 1.5, label = "Temperatura Média")
sns.lineplot(x = clima.index, y = clima["tmin"],  ax = axs[0],
				color = "darkblue", linewidth = 1.5, label = "Temperatura Mínima")
sns.lineplot(x = clima.index, y = clima["heat_index"],  ax = axs[0],
				color = "red", linewidth = 1.8, linestyle = ":", label = "Heat Index")
sns.lineplot(x = clima.index, y = clima["wind_chill"],  ax = axs[0],
				color = "darkblue", linewidth = 1.8, linestyle = ":", label = "Wind Chill")
axs[0].set_ylabel("Temperaturas (C)")
axs[0].legend(loc = "lower left")#, frameon = True, facecolor = "white", edgecolor = "black")
"""
leg = axs[0].legend(loc="lower left")
leg.get_frame().set_facecolor("white")  
leg.get_frame().set_alpha(1.0)          
leg.get_frame().set_edgecolor("black")
"""
#axs[0].set_ylim(0, 45)
#sns.set()
ax2 = axs[0].twinx()
sns.lineplot(x = clima.index, y = clima["obito"], ax = ax2,
				color = "purple", linewidth = 1, linestyle = "--", label = "Óbitos Cardiovasculares")
ax2.fill_between(clima.index, clima["obito"], color = "purple", alpha = 0.3)
ax2.set_ylabel("Óbitos Cardiovasculares")
ax2.legend(loc = "lower right")
"""
#sns.set_theme(style="whitegrid")  # "ticks" Ou 'whitegrid', mas EVITE 'darkgrid'
sns.lineplot(x = clima.index, y = clima["focos"],  ax = ax2,
				color = "darkgreen", linewidth = 1, linestyle = ":", label = "Focos de _Aedes_ sp.")
ax2.fill_between(clima.index, clima["focos"], color = "darkgreen", alpha = 0.35)
ax2.set_ylabel("Focos de _Aedes_ sp.")
ax2.legend(loc = "upper left")
"""
axs[1].set_facecolor("honeydew") #.gcf()
ax3 = axs[1].twinx()#.set_facecolor("honeydew")
sns.barplot(x = clima["dia"], y = clima["prec"],  ax = ax3,
				color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
ax3.set_ylabel("Precipitação (mm)")
ax3.legend(loc = "upper right")
sns.lineplot(x = clima.index, y = clima["urmax"],  ax = axs[1],
				color = "red", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Máxima")
sns.lineplot(x = clima.index, y = clima["umidade"],  ax = axs[1],
				color = "orange", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa")
sns.lineplot(x = clima.index, y = clima["urmin"],  ax = axs[1],
				color = "darkblue", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Mínima")
axs[1].set_ylabel("Umidade Relativa (%)")
axs[1].set_ylim(50, 105)
axs[1].legend(loc = "upper left")
axs[1].grid(False)
meses = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
nomes_meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
plt.xticks(meses, nomes_meses)
axs[1].set_xlabel("Ciclo Anual (dias)")
#fig.suptitle(f"CASOS DE DENGUE, FOCOS DE _Aedes_ sp., TEMPERATURAS (MÍNIMA, MÉDIA E MÁXIMA) E PRECIPITAÇÃO.\nSAZONALIDADE POR MÉDIAS SEMANAIS PARA O MUNICÍPIO DE clima, SANTA CATARINA.")
nome_arquivo = f"esbmet25_distribuicao_sazonal_subplots_diario.pdf"
caminho_esbmet = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/esbmet/"
#if _SALVAR == True:
os.makedirs(caminho_esbmet, exist_ok = True)
plt.savefig(f'{caminho_esbmet}{nome_arquivo}', format = "pdf", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
print(f"""\n{green}SALVO COM SUCESSO!\n
{cyan}ENCAMINHAMENTO: {caminho_esbmet}\n
NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
#if _VISUALIZAR == True:
print(f"\n{cyan}Visualizando:\n{caminho_esbmet}{nome_arquivo}\n{reset}")
plt.show()
#sys.exit()

### Pré-Processamento (Climatologia MÊS)
clima_mes = clima.drop(columns = "dia.1")
clima_mes["data"] = clima_mes["dia"].apply(lambda x: datetime(2020, 1, 1) + timedelta(days=int(x) - 1))
clima_mes["mes"] = clima_mes["data"].dt.month
clima_mes.drop(columns = ["dia", "data"], inplace = True)
clima_mes = clima_mes.groupby("mes").mean(numeric_only = True)


print(f"\n{green}SAZONALIDADE MENSAL\n{reset}{clima_mes}\n")
#sys.exit()

### Visualização Gráfica (SAZONALIDADE MENSAL)
fig, axs = plt.subplots(2, 1, figsize = (12, 6), layout = "tight", frameon = False,  sharex = True)
axs[0].set_facecolor("honeydew") #.gcf()
axs[0].legend(loc = "upper center")
axs[0].legend(loc = "lower left")
sns.lineplot(x = clima_mes.index, y = clima_mes["tmax"],  ax = axs[0],
				color = "red", linewidth = 1.5, label = "Temperatura Máxima")
sns.lineplot(x = clima_mes.index, y = clima_mes["temp"],  ax = axs[0],
				color = "orange", linewidth = 1.5, label = "Temperatura Média")
sns.lineplot(x = clima_mes.index, y = clima_mes["tmin"],  ax = axs[0],
				color = "darkblue", linewidth = 1.5, label = "Temperatura Mínima")
sns.lineplot(x = clima_mes.index, y = clima_mes["heat_index"],  ax = axs[0],
				color = "red", linewidth = 1.8, linestyle = ":", label = "Heat Index")
sns.lineplot(x = clima_mes.index, y = clima_mes["wind_chill"],  ax = axs[0],
				color = "darkblue", linewidth = 1.8, linestyle = ":", label = "Wind Chill")
axs[0].set_ylabel("Temperaturas (C)")
ax2 = axs[0].twinx()
sns.lineplot(x = clima_mes.index, y = clima_mes["obito"], ax = ax2,
				color = "purple", linewidth = 1, linestyle = "--", label = "Óbitos Cardiovasculares")
ax2.fill_between(clima_mes.index, clima_mes["obito"], color = "purple", alpha = 0.3)
ax2.set_ylabel("Óbitos Cardiovasculares")
ax2.legend(loc = "lower right")
axs[1].set_facecolor("honeydew")
ax3 = axs[1].twinx()
ax3.bar(clima_mes.index, clima_mes["prec"], color = "royalblue", alpha = 0.7, label = "Precipitação")
ax3.set_ylabel("Precipitação (mm)")
ax3.legend(loc = "upper right")
"""
sns.barplot(x = clima_mes.index, y = clima_mes["prec"],  ax = ax3,
				color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
sns.barplot(x = clima_mes.index.astype(str), ci = None, y = clima_mes["prec"],  ax = ax3,
				color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
"""

sns.lineplot(x = clima_mes.index, y = clima_mes["urmax"],  ax = axs[1],
				color = "red", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Máxima")
sns.lineplot(x = clima_mes.index, y = clima_mes["umidade"],  ax = axs[1],
				color = "orange", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa")
sns.lineplot(x = clima_mes.index, y = clima_mes["urmin"],  ax = axs[1],
				color = "darkblue", linewidth = 1.5, linestyle = "--", label = "Umidade Relativa Mínima")
axs[1].set_ylabel("Umidade Relativa (%)")
axs[1].set_ylim(50, 105)
axs[1].legend(loc = "upper left")
axs[1].grid(False)
meses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
nomes_meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
plt.xticks(meses, nomes_meses)
axs[1].set_xlabel("Ciclo Anual (meses)")
#fig.suptitle(f"CASOS DE DENGUE, FOCOS DE _Aedes_ sp., TEMPERATURAS (MÍNIMA, MÉDIA E MÁXIMA) E PRECIPITAÇÃO.\nSAZONALIDADE POR MÉDIAS SEMANAIS PARA O MUNICÍPIO DE clima_mes, SANTA CATARINA.")
nome_arquivo = f"esbmet25_distribuicao_sazonal_subplots_mensal.pdf"
caminho_esbmet = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/esbmet/"
#if _SALVAR == True:
os.makedirs(caminho_esbmet, exist_ok = True)
plt.savefig(f'{caminho_esbmet}{nome_arquivo}', format = "pdf", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
print(f"""\n{green}SALVO COM SUCESSO!\n
{cyan}ENCAMINHAMENTO: {caminho_esbmet}\n
NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
#if _VISUALIZAR == True:
print(f"\n{cyan}Visualizando:\n{caminho_esbmet}{nome_arquivo}\n{reset}")
plt.show()
#sys.exit()



