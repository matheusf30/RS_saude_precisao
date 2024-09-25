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
                                                       ##### FILTRAR AS 5 CID +FREQUENTES
############################################################ PLOTAR 1 GRÁFICO/ÍNDICE
"""
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
print(f"\nbio\n{bio}\n")
print(f'\nbio["causa"].value_counts()\n{bio["causa"].value_counts()}\n')
# Top 50
top50 = bio["causa"].value_counts().head(50)
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
top50.plot.bar()
plt.title("50 PRINCIPAIS CAUSAS DE ÓBITOS CARDIOVASCULARES (CID).\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xticks(np.arange(50), bio["causa"].value_counts().head(50),  rotation = 70)
plt.xlabel("CID")
plt.ylabel("Quantidade")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices_top/"
	os.makedirs(caminho_indice, exist_ok = True)
	nome_arquivo = "distribuicao_top50_porto_alegre.pdf"
	plt.savefig(f'{caminho_indice}{nome_arquivo}',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição das 50 principais causas de óbitos cardiovasculares do município de Porto Alegre.{reset}")
	plt.show()
# Top 5
top5 = bio["causa"].value_counts().head(20)
print(top5.index)
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
top5.plot.bar()
plt.title("5 PRINCIPAIS CAUSAS DE ÓBITOS CARDIOVASCULARES (CID).\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xticks(np.arange(5), bio["causa"].value_counts().head(5),  rotation = "horizontal")
plt.xlabel("CID")
plt.ylabel("Quantidade")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices_top/"
	os.makedirs(caminho_indice, exist_ok = True)
	nome_arquivo = "distribuicao_top5_porto_alegre.pdf"
	plt.savefig(f'{caminho_indice}{nome_arquivo}',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição das 5 principais causas de óbitos cardiovasculares do município de Porto Alegre.{reset}")
	plt.show()
sys.exit()
# Filtro Top5
top5 = bio["causa"].value_counts().head(5).index
bio = bio[bio["causa"].isin(top5)]
print(f"\nbio = bio[bio['causa'].isin(top5)]\n{bio}\n")
bio.drop(columns=["CODMUNRES", "diaobito", "mesobito", "anoobito"], inplace = True)
bio = bio[["data", "obito", "sexo", "idade", "causa"]].sort_values(by = "data")
#bio = bio.groupby(by = ["data"])["obito"].sum()
total = bio.groupby(by = ["data"])["obito"].sum()
#sexo = bio.groupby(by = ["data", "sexo"])["obito"].sum()
#idade = bio.groupby(by = ["data", "idade"])["obito"].sum()
causa = bio.groupby(by = ["data", "causa"])["obito"].sum()
causa = causa.reset_index()
causa = causa.rename(columns={causa.columns[-1]: "obito"})
print(f"\n{green}bio{reset}\n{bio}\n")
print(f"\n{green}total{reset}\n{total}\n")
print(f"\n{green}causa{reset}\n{causa}\n")
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.histplot(data = causa, x = "obito", hue = "causa", multiple = "dodge", shrink = 10) #y = "obito"
plt.title("5 PRINCIPAIS CAUSAS DE ÓBITOS CARDIOVASCULARES (CID).\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
#plt.xticks(np.arange(5), bio["causa"].value_counts().head(50),  rotation = 70)
plt.xlabel("Óbitos Diários")
plt.ylabel("Quantidade")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices_top/"
	os.makedirs(caminho_indice, exist_ok = True)
	nome_arquivo = "serie_top5_porto_alegre.csv"
	causa.to_csv(f'{caminho_indice}{nome_arquivo}', index = False)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
	nome_arquivo = "distribuicao_dia_top5_porto_alegre.pdf"
	plt.savefig(f'{caminho_indice}{nome_arquivo}',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição das 5 principais causas de óbitos cardiovasculares do município de Porto Alegre.{reset}")
	plt.show()
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.histplot(data = causa, x = "obito", hue = "causa", multiple = "dodge", shrink = 10) #y = "obito"
plt.title("5 PRINCIPAIS CAUSAS DE ÓBITOS CARDIOVASCULARES (CID).\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
#plt.xticks(np.arange(5), bio["causa"].value_counts().head(50),  rotation = 70)
plt.xlabel("Óbitos Diários")
plt.ylabel("Quantidade")
#sys.exit()

# Oscilação Anual
total.reset_index(inplace = True)
timeindex = total.copy()
timeindex = timeindex.set_index("data")
timeindex["dia"] = timeindex.index.dayofyear
#timeindex["mes_dia"] = timeindex.index.to_period('M').astype(str) + '-' + timeindex.index.day.astype(str)
print(f"\n{green}timeindex\n{reset}{timeindex}\n")
print("~"*80)
print(f"{green}\ntimeindex.info(){reset}\n{timeindex.info()}\n")
print("="*80)
sys.exit()
# Média Diária
media_dia = timeindex.groupby("dia").mean().round(2)
#media_dia.reset_index(inplace = True)
print(f"\n{green}media_dia{reset}\n{media_dia}\n")
print(f"\n{green}media_dia.index{reset}\n{media_dia.index}\n")
# Desvio Padrão Diário
desvio_dia = timeindex.groupby("dia").std().round(2)
#desvio_dia.reset_index(inplace = True)
print(f"\n{green}desvio_dia{reset}\n{desvio_dia}\n")
print(f"\n{green}desvio_dia.index{reset}\n{desvio_dia.index}\n")
# Percentil 25% Diário
p25_dia = timeindex.groupby("dia").quantile(0.25).round(2)
#p25_dia.reset_index(inplace = True)
print(f"\n{green}p25_dia{reset}\n{p25_dia}\n")
print(f"\n{green}p25_dia.index{reset}\n{p25_dia.index}\n")
# Percentil 75% Diário
p75_dia = timeindex.groupby("dia").quantile(0.75).round(2)
#p75_dia.reset_index(inplace = True)
print(f"\n{green}p75_dia{reset}\n{p75_dia}\n")
print(f"\n{green}p75_dia.index{reset}\n{p75_dia.index}\n")
sys.exit()

# Índices de Alta Mortalidade
# 1 >> MEAN + 1.96 * STD
iam1 = media_dia + (1.96 * desvio_dia)
print(f"\n{green}iam1{reset}\n{iam1}")

# 2 >> MEAN + 3 * STD
iam2 = media_dia + 3 * desvio_dia
print(f"\n{green}iam2{reset}\n{iam2}")

# 3 >> P75
iam3 = p75_dia.copy()
print(f"\n{green}iam3{reset}\n{iam3}")

# 4 >> P75 + 1.5 * (P75-P25)
iam4 = p75_dia + 1.5 * (p75_dia - p25_dia)
print(f"\n{green}iam4{reset}\n{iam4}")

# Filtrando Índices da Série Histórica
# IAM1
iam1.reset_index(inplace = True)
iam1.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}iam1\n{reset}{iam1}\n")
timeindex1 = timeindex.copy()
timeindex1.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}timeindex1\n{reset}{timeindex1}\n")
serie_total1 = timeindex1.merge(iam1, left_on = "dia", how = "left", suffixes = ("", "_iam1"), right_index = True)
serie_iam1 = pd.DataFrame(index = timeindex1.index)
print(f"\n{green}serie_total1\n{reset}{serie_total1}\n")
for coluna in timeindex1.columns:
	if coluna in serie_total1.columns:
		x_coluna = f"{coluna}_iam1"
		if x_coluna in serie_total1.columns:
			#serie_iam1[coluna] = timeindex1[coluna] - serie_total[x_coluna]
			#serie_iam1[coluna] = timeindex1[coluna].apply(lambda x: timeindex1[coluna] if timeindex1[coluna] > serie_total[x_coluna] else 0)
			serie_iam1[coluna] = timeindex1[coluna].where(timeindex1[coluna] > serie_total1[x_coluna], 0)
		else:
			print(f"\n{red}Coluna {x_coluna} não encontrada na série!{reset}\n")
	else:
		print(f"\n{red}Coluna {coluna} não encontrada no timeindex!{reset}\n")
serie_iam1.drop(columns = "dia", inplace = True)
print(f"\n{green}serie_iam1\n{reset}{serie_iam1}\n")
print(f"\n{green}serie_iam1.columns\n{reset}{serie_iam1.columns}\n")
print(f"\n{green}(serie_iam1 == 0).sum()\n{reset}{(serie_iam1 == 0).sum()}\n")
print(f"\n{green}len(serie_iam1) - (serie_iam1 == 0).sum()\n{reset}{len(serie_iam1) - (serie_iam1 == 0).sum()}\n")
print(f"\n{green}serie_iam1.describe()\n{reset}{serie_iam1.describe()}\n")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	serie_iam1.reset_index(inplace = True)
	serie_iam1.to_csv(f'{caminho_indice}serie_IAM1_porto_alegre.csv', index = False)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM1_porto_alegre.csv{reset}\n""")
# IAM2
iam2.reset_index(inplace = True)
iam2.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}iam2\n{reset}{iam2}\n")
timeindex2 = timeindex.copy()
timeindex2.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}timeindex2\n{reset}{timeindex2}\n")
serie_total2 = timeindex2.merge(iam2, left_on = "dia", how = "left", suffixes = ("", "_iam2"), right_index = True)
serie_iam2 = pd.DataFrame(index = timeindex2.index)
print(f"\n{green}serie_total2\n{reset}{serie_total2}\n")
for coluna in timeindex2.columns:
	if coluna in serie_total2.columns:
		x_coluna = f"{coluna}_iam2"
		if x_coluna in serie_total2.columns:

			serie_iam2[coluna] = timeindex2[coluna].where(timeindex2[coluna] > serie_total2[x_coluna], 0)
		else:
			print(f"\n{red}Coluna {x_coluna} não encontrada na série!{reset}\n")
	else:
		print(f"\n{red}Coluna {coluna} não encontrada no timeindex!{reset}\n")
serie_iam2.drop(columns = "dia", inplace = True)
print(f"\n{green}serie_iam2\n{reset}{serie_iam2}\n")
print(f"\n{green}serie_iam2.columns\n{reset}{serie_iam2.columns}\n")
print(f"\n{green}(serie_iam2 == 0).sum()\n{reset}{(serie_iam2 == 0).sum()}\n")
print(f"\n{green}len(serie_iam2) - (serie_iam2 == 0).sum()\n{reset}{len(serie_iam2) - (serie_iam2 == 0).sum()}\n")
print(f"\n{green}serie_iam2.describe()\n{reset}{serie_iam2.describe()}\n")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	serie_iam2.reset_index(inplace = True)
	serie_iam2.to_csv(f'{caminho_indice}serie_IAM2_porto_alegre.csv', index = False)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM2_porto_alegre.csv{reset}\n""")
# IAM3
iam3.reset_index(inplace = True)
iam3.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}iam3\n{reset}{iam3}\n")
timeindex3 = timeindex.copy()
timeindex3.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}timeindex3\n{reset}{timeindex3}\n")
serie_total3 = timeindex3.merge(iam3, left_on = "dia", how = "left", suffixes = ("", "_iam3"), right_index = True)
serie_iam3 = pd.DataFrame(index = timeindex3.index)
print(f"\n{green}serie_total3\n{reset}{serie_total3}\n")
for coluna in timeindex3.columns:
	if coluna in serie_total3.columns:
		x_coluna = f"{coluna}_iam3"
		if x_coluna in serie_total3.columns:
			serie_iam3[coluna] = timeindex3[coluna].where(timeindex3[coluna] > serie_total3[x_coluna], 0)
		else:
			print(f"\n{red}Coluna {x_coluna} não encontrada na série!{reset}\n")
	else:
		print(f"\n{red}Coluna {coluna} não encontrada no timeindex!{reset}\n")
serie_iam3.drop(columns = "dia", inplace = True)
print(f"\n{green}serie_iam3\n{reset}{serie_iam3}\n")
print(f"\n{green}serie_iam3.columns\n{reset}{serie_iam3.columns}\n")
print(f"\n{green}(serie_iam3 == 0).sum()\n{reset}{(serie_iam3 == 0).sum()}\n")
print(f"\n{green}len(serie_iam3) - (serie_iam3 == 0).sum()\n{reset}{len(serie_iam3) - (serie_iam3 == 0).sum()}\n")
print(f"\n{green}serie_iam3.describe()\n{reset}{serie_iam3.describe()}\n")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	serie_iam3.reset_index(inplace = True)
	serie_iam3.to_csv(f'{caminho_indice}serie_IAM3_porto_alegre.csv', index = False)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM3_porto_alegre.csv{reset}\n""")
# IAM4
iam4.reset_index(inplace = True)
iam4.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}iam4\n{reset}{iam4}\n")
timeindex4 = timeindex.copy()
timeindex4.drop(columns = ['tmin', 'temp', 'tmax', 'amplitude_t',
					'urmin', 'umidade', 'urmax', 'prec',
					'pressao', 'ventodir', 'ventovel'], inplace = True)
print(f"\n{green}timeindex4\n{reset}{timeindex4}\n")
serie_total4 = timeindex4.merge(iam4, left_on = "dia", how = "left", suffixes = ("", "_iam4"), right_index = True)
serie_iam4 = pd.DataFrame(index = timeindex4.index)
print(f"\n{green}serie_total4\n{reset}{serie_total4}\n")
for coluna in timeindex4.columns:
	if coluna in serie_total4.columns:
		x_coluna = f"{coluna}_iam4"
		if x_coluna in serie_total4.columns:
			serie_iam4[coluna] = timeindex4[coluna].where(timeindex4[coluna] > serie_total4[x_coluna], 0)
		else:
			print(f"\n{red}Coluna {x_coluna} não encontrada na série!{reset}\n")
	else:
		print(f"\n{red}Coluna {coluna} não encontrada no timeindex!{reset}\n")
serie_iam4.drop(columns = "dia", inplace = True)
print(f"\n{green}serie_iam4\n{reset}{serie_iam4}\n")
print(f"\n{green}serie_iam4.columns\n{reset}{serie_iam4.columns}\n")
print(f"\n{green}(serie_iam4 == 0).sum()\n{reset}{(serie_iam4 == 0).sum()}\n")
print(f"\n{green}len(serie_iam4) - (serie_iam4 == 0).sum()\n{reset}{len(serie_iam4) - (serie_iam4 == 0).sum()}\n")
print(f"\n{green}serie_iam4.describe()\n{reset}{serie_iam4.describe()}\n")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	serie_iam4.reset_index(inplace = True)
	serie_iam4.to_csv(f'{caminho_indice}serie_IAM4_porto_alegre.csv', index = False)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM4_porto_alegre.csv{reset}\n""")

### Visualização Gráfica dos Índices
# Todos Juntos
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = timeindex.index, y = timeindex["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.5 )
sns.scatterplot(x = timeindex.index, y = serie_iam1["obito"],
				color = "blue", marker = "D", alpha = 0.7,
				label = "IAM1 = média diária + 1,6 * desvio padrão diário")
sns.scatterplot(x = timeindex.index, y = serie_iam2["obito"],
				color = "green", marker = "o",# alpha = 0.7,
				label = "IAM2 = média diária + 3 * desvio padrão diário")
sns.scatterplot(x = timeindex.index, y = serie_iam3["obito"],
				color = "purple", marker = "x", alpha = 0.7,
				label = "IAM3 = percentil 75")
sns.scatterplot(x = timeindex.index, y = serie_iam4["obito"],
				color = "red", marker = ".", alpha = 0.7,
				label = "IAM4 = terceiro quartil + 1,5 * intervalo interquartil")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ÍNDICES DE ALTA MORTALIDADE.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	plt.savefig(f'{caminho_indice}serie_IAM_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e índices de alta mortalidade do município de Porto Alegre.{reset}")
	plt.show()
# Índice 1
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = timeindex.index, y = timeindex["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.7 )
sns.scatterplot(x = timeindex.index, y = serie_iam1["obito"],
				color = "blue", marker = "o", alpha = 0.7,
				label = "IAM1 = média diária + 1,6 * desvio padrão diário")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ÍNDICES DE ALTA MORTALIDADE.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	plt.savefig(f'{caminho_indice}serie_IAM1_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM1_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e índice 1 de alta mortalidade do município de Porto Alegre.{reset}")
	plt.show()
# Índice 2
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = timeindex.index, y = timeindex["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.7 )
sns.scatterplot(x = timeindex.index, y = serie_iam2["obito"],
				color = "green", marker = "o",# alpha = 0.7,
				label = "IAM2 = média diária + 3 * desvio padrão diário")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ÍNDICES DE ALTA MORTALIDADE.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	plt.savefig(f'{caminho_indice}serie_IAM2_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM2_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e índice 2 de alta mortalidade do município de Porto Alegre.{reset}")
	plt.show()
# Índice 3
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = timeindex.index, y = timeindex["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.7 )
sns.scatterplot(x = timeindex.index, y = serie_iam3["obito"],
				color = "purple", marker = "o", alpha = 0.7,
				label = "IAM3 = percentil 75")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ÍNDICES DE ALTA MORTALIDADE.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	plt.savefig(f'{caminho_indice}serie_IAM3_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM3_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e índice 3 de alta mortalidade do município de Porto Alegre.{reset}")
	plt.show()
# Índice 4
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = timeindex.index, y = timeindex["obito"], label = "Óbito",
				color = "black", linewidth = 1, alpha = 0.7 )
sns.scatterplot(x = timeindex.index, y = serie_iam4["obito"],
				color = "red", marker = "o", alpha = 0.7,
				label = "IAM4 = terceiro quartil + 1,5 * intervalo interquartil")
plt.legend()
plt.title("DISTRIBUIÇÃO DE ÓBITOS CARDIOVASCULARES E ÍNDICES DE ALTA MORTALIDADE.\nMUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares")
if _SALVAR == "True":
	caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	os.makedirs(caminho_indice, exist_ok = True)
	plt.savefig(f'{caminho_indice}serie_IAM4_porto_alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{green}SALVO COM SUCESSO!
	{cyan}ENCAMINHAMENTO: {caminho_indice}
	NOME DO ARQUIVO: serie_IAM4_porto_alegre.pdf{reset}\n""")
if _VISUALIZAR == "True":
	print(f"{green}Exibindo a distribuição de óbitos e índice 4 de alta mortalidade do município de Porto Alegre.{reset}")
	plt.show()

