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

##### Padrão ANSI ###############################################################
ansi = {"bold" : "\033[1m", "red" : "\033[91m",
        "green" : "\033[92m", "yellow" : "\033[33m",
        "blue" : "\033[34m", "magenta" : "\033[35m",
        "cyan" : "\033[36m", "white" : "\033[37m", "reset" : "\033[0m"}
#################################################################################

if _LOCAL == "GH":
	caminho_dados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/dados/"
	caminho_resultados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/resultados/porto_alegre/"
	caminho_modelos = "https://github.com/matheusf30/RS_saude_precisao/tree/main/modelos/"
elif _LOCAL ==  "SIFAPSC":
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
biometeoro = "biometeoro_PORTO_ALEGRE.csv"
verao = "biometeoro_verao_PORTO_ALEGRE.csv"
inverno = "biometeoro_inverno_PORTO_ALEGRE.csv"

### Abrindo Arquivo
biometeoro = pd.read_csv(f"{caminho_dados}{biometeoro}", low_memory = False)
verao = pd.read_csv(f"{caminho_dados}{verao}", low_memory = False)
inverno = pd.read_csv(f"{caminho_dados}{inverno}", low_memory = False)

### Pré-Processamento

# BIOMETEORO
print(80*"=")
print("\n\nBIOMEOTEORO\n\n")
print(biometeoro, biometeoro.info())
print(80*"=")
print("\n\nVERÃO\n\n")
print(verao, verao.info())
print(80*"=")
print("\n\nINVERNO\n\n")
print(inverno, inverno.info())
print(80*"=")
print("\nbiometeoro.iloc[-365:,:]\n", biometeoro.iloc[-365:,:])

#sys.exit()



# Tratando Sazonalidade
timeindex = biometeoro.copy()
timeindex["data"] = pd.to_datetime(timeindex["data"])
timeindex = timeindex.set_index("data")
timeindex["dia"] = timeindex.index.dayofyear
print("\ntimeindex\n", timeindex,"~"*80, timeindex.info())
print("="*80)
#sys.exit()
media_dia = timeindex.groupby("dia").mean().round(2)
media_dia.reset_index(inplace = True)
print(media_dia)
print(media_dia.index)
#sys.exit()
plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = media_dia["dia"], y = media_dia["obito"],
				color = "black", linewidth = 1, label = "Óbito")
sns.lineplot(x = media_dia["dia"], y = media_dia["tmin"],
				color = "darkblue", linewidth = 1, label = "Temperatura Mínima")
sns.lineplot(x = media_dia["dia"], y = media_dia["tmax"],
				color = "red", linewidth = 1, label = "Temperatura Máxima")
plt.title("DISTRIBUIÇÃO DE TEMPERATURA MÍNIMA, TEMPERATURA MÁXIMA E ÓBITOS CARDIOVASCULARES.\nMÉDIAS ANUAIS PARA O MUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares X Temperaturas (C)")

if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/descritiva/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	media_dia.to_csv(f"{caminho_dados}climatologia.csv", index = True)
	print(f"\n{ansi['green']}SALVO COM SUCESSO!\nCLIMATOLOGIA.csv\n{ansi['reset']}")
	plt.savefig(f'{caminho_correlacao}distribuicao_medianual_tmin_tmax_obitos_Porto_Alegre.pdf',
				format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: {caminho_correlacao}distribuicao_medianual_tmin_tmax_obitos_Porto_Alegre.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a distribuição de tmin, tmax e óbitos do município de Porto Alegre. Média Anual. {ansi['reset']}")
	plt.show()

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

plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
sns.lineplot(x = sem_sazonal.index, y = sem_sazonal["tmin"],
				color = "darkblue", linewidth = 1, label = "Temperatura Mínima")
sns.lineplot(x = sem_sazonal.index, y = sem_sazonal["tmax"],
				color = "red", linewidth = 1, label = "Temperatura Máxima", alpha = 0.7)#, linewidth = 3
sns.lineplot(x = sem_sazonal.index, y = sem_sazonal["obito"],
				color = "black", linewidth = 1, label = "Óbito", alpha = 0.7)
plt.title("DISTRIBUIÇÃO DE TEMPERATURA MÍNIMA, TEMPERATURA MÁXIMA E ÓBITOS CARDIOVASCULARES.\nSEM SAZONALIDADE AO LONGO DOS ANOS PARA O MUNICÍPIO DE PORTO ALEGRE, RIO GRANDE DO SUL.")
plt.xlabel("Série Histórica (Observação Diária)")
plt.ylabel("Número de Óbitos Cardiovasculares X Temperaturas (C)")
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/descritiva/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	sem_sazonal.to_csv(f"{caminho_dados}anomalia.csv", index = True)
	print(f"\n{ansi['green']}SALVO COM SUCESSO!\nANOMALIA.csv\n{ansi['reset']}")
	plt.savefig(f'{caminho_correlacao}distribuicao_seriehistorica_semsazonal_tmin_tmax_obitos_Porto_Alegre.pdf',
				format = "pdf", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: {caminho_correlacao}distribuicao_seriehistorica_semsazonal_tmin_tmax_obitos_Porto_Alegre.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a distribuição de tmin, tmax e óbitos do município de Porto Alegre. Série histórica.{ansi['reset']}")
	plt.show()
#sys.exit()
# Verificando Tendência
colunas = ["obito", "heat_index", "wind_chill",
			"tmin", "temp", "tmax",
			"amplitude_t","urmin", "umidade", "urmax",
			"prec", "pressao", "ventodir", "ventovel"]
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
if _SALVAR == "True":
	anomalia_estacionaria.to_csv(f"{caminho_dados}anomalia_estacionaria.csv", index = True)

#sys.exit()
#################################################################################
### Correlações
#################################################################################
lista_METODO = ["pearson", "spearman"]#, "pearson", "spearman"

colunas_r = ["heat_index", "wind_chill",
			"tmin", "temp", "tmax",
			"amplitude_t","urmin", "umidade", "urmax",
			"prec", "pressao", "ventodir", "ventovel"]
_retroceder = [3, 6, 9]

# Correlações Dados Brutos Durante Invernos
# Até 3 dias retroagidos
inverno_set = inverno.copy()
inverno_set = inverno_set.drop(columns = "data")
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		inverno_set[f"{c_r}_r{_r}"] = inverno_set[f"{c_r}"].shift(-_r)
inverno_set.dropna(inplace = True)
print("\ninverno_set\n", inverno_set)
for _METODO in lista_METODO:
	correlacao_dataset = inverno_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS ENTRE ABRIL E SETEMBRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos entre abril e setembro do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
		plt.show()
# Até 6 dias retroagidos
inverno_set = inverno.copy()
inverno_set = inverno_set.drop(columns = "data")
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		inverno_set[f"{c_r}_r{_r}"] = inverno_set[f"{c_r}"].shift(-_r)
inverno_set.dropna(inplace = True)
print("\ninverno_set\n", inverno_set)
for _METODO in lista_METODO:
	correlacao_dataset = inverno_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS DURANTE ENTRE ABRIL E SETEMBRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO} de dados brutos entre abril e setembro do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
		plt.show()
# Até 9 dias retroagidos
inverno_set = inverno.copy()
inverno_set = inverno_set.drop(columns = "data")
for _r in range(_retroceder[1] + 1, _retroceder[2] + 1):
	for c_r in colunas_r:
		inverno_set[f"{c_r}_r{_r}"] = inverno_set[f"{c_r}"].shift(-_r)
inverno_set.dropna(inplace = True)
print("\ninverno_set\n", inverno_set)
for _METODO in lista_METODO:
	correlacao_dataset = inverno_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS ENTRE ABRIL E SETEMBRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_inverno_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos entre abril e setembro do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
		plt.show()

# Correlações Dados Brutos Durante Verões
# Até 3 dias retroagidos
verao_set = verao.copy()
verao_set = verao_set.drop(columns = "data")
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		verao_set[f"{c_r}_r{_r}"] = verao_set[f"{c_r}"].shift(-_r)
verao_set.dropna(inplace = True)
print("\nverao_set\n", verao_set)
for _METODO in lista_METODO:
	correlacao_dataset = verao_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS ENTRE DEZEMBRO E FEVEREIRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos entre dezembro e fevereiro do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
		plt.show()
# Até 6 dias retroagidos
verao_set = verao.copy()
verao_set = verao_set.drop(columns = "data")
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		verao_set[f"{c_r}_r{_r}"] = verao_set[f"{c_r}"].shift(-_r)
verao_set.dropna(inplace = True)
print("\nverao_set\n", verao_set)
for _METODO in lista_METODO:
	correlacao_dataset = verao_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS DURANTE ENTRE DEZEMBRO E FEVEREIRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO} de dados brutos entre dezembro e fevereiro do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
		plt.show()
# Até 9 dias retroagidos
verao_set = verao.copy()
verao_set = verao_set.drop(columns = "data")
for _r in range(_retroceder[1] + 1, _retroceder[2] + 1):
	for c_r in colunas_r:
		verao_set[f"{c_r}_r{_r}"] = verao_set[f"{c_r}"].shift(-_r)
verao_set.dropna(inplace = True)
print("\nverao_set\n", verao_set)
for _METODO in lista_METODO:
	correlacao_dataset = verao_set.corr(method = f"{_METODO}")
	fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
	filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
	sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
				vmin = -1, vmax = 1, linewidth = 0.5,
				mask = filtro, annot_kws={"size": 6})
	fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS ENTRE DEZEMBRO E FEVEREIRO.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.",
					weight = "bold", size = "medium")
	ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
	ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
	if _SALVAR == "True":
		caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
		os.makedirs(caminho_correlacao, exist_ok = True)
		plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
		print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
		{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
		NOME DO ARQUIVO: matriz_correlacao_{_METODO}_verao_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos entre dezembro e fevereiro do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
		plt.show()
#sys.exit()

# Anomalias Estacionárias
# Até 3 dias retroagidos
anomalia_estacionaria_set = anomalia_estacionaria.copy()
#anomalia_estacionaria_set = anomalia_estacionaria_set.drop(columns = "data")
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de anomalias estacionárias do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 6 dias retroagidos
anomalia_estacionaria_set = anomalia_estacionaria.copy()
#anomalia_estacionaria_set = anomalia_estacionaria_set.drop(columns = "data")
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO} de anomalias estacionárias do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 9 dias retroagidos
anomalia_estacionaria_set = anomalia_estacionaria.copy()
#anomalia_estacionaria_set = anomalia_estacionaria_set.drop(columns = "data")
for _r in range(_retroceder[1] + 1, _retroceder[2] + 1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de anomalias estacionárias do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
	plt.show()

# Anomalia Estacionária do Último Ano (2022)
print("\nanomalia_estacionaria.iloc[-365:,:]\n", anomalia_estacionaria.iloc[-365:,:])
# Até 3 dias retroagidos
ultimo_ano = anomalia_estacionaria.iloc[-365:,:]
anomalia_estacionaria_set = ultimo_ano.copy()
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set (2022)\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria2022_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio2022_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de anomalias estacionárias em 2022 do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 6 dias retroagidos
anomalia_estacionaria_set = ultimo_ano.copy()
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set (2022)\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria2022_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio2022_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO} de anomalias estacionárias em 2022 do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 9 dias retroagidos
anomalia_estacionaria_set = ultimo_ano.copy()
for _r in range(_retroceder[1] + 1, _retroceder[2] + 1):
	for c_r in colunas_r:
		anomalia_estacionaria_set[f"{c_r}_r{_r}"] = anomalia_estacionaria_set[f"{c_r}"].shift(-_r)
anomalia_estacionaria_set.dropna(inplace = True)
print("\nanomalia_estacionaria_set (2022)\n", anomalia_estacionaria_set)
correlacao_dataset = anomalia_estacionaria_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE ANOMALIAS ESTACIONÁRIAS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_anomaliaestacionaria2022_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_anomaestacio2022_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de anomalias estacionárias em 2022 do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
	plt.show()

# Dados Brutos
# Até 3 dias retroagidos
biometeoro2 = biometeoro.set_index("data")
biometeoro_set = biometeoro2.copy()
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 6 dias retroagidos
biometeoro_set = biometeoro2.copy()
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 9 dias retroagidos
biometeoro_set = biometeoro2.copy()
for _r in range(_retroceder[1] + 1, _retroceder[2] +1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
	plt.show()
sys.exit()
# Dados Brutos de 2022
# Até 3 dias retroagidos
ultimo_ano = biometeoro.iloc[-365:,:].set_index("data")
biometeoro_set = ultimo_ano.copy()
for _r in range(1, _retroceder[0] +1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set (2022)\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[0]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[0]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[0]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos em 2022 do município de Porto Alegre com até {_retroceder[0]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 6 dias retroagidos
biometeoro_set = ultimo_ano.copy()
for _r in range(_retroceder[0] + 1, _retroceder[1] + 1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set (2022)\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[1]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[1]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[1]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos em 2022 do município de Porto Alegre com até {_retroceder[1]} dias retroagidos. {ansi['reset']}")
	plt.show()
# Até 9 dias retroagidos
biometeoro_set = ultimo_ano.copy()
for _r in range(_retroceder[1] + 1, _retroceder[2] +1):
	for c_r in colunas_r:
		biometeoro_set[f"{c_r}_r{_r}"] = biometeoro_set[f"{c_r}"].shift(-_r)
biometeoro_set.dropna(inplace = True)
print("\nbiometeoro_set (2022)\n", biometeoro_set)
correlacao_dataset = biometeoro_set.corr(method = f"{_METODO}")
fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral",
			vmin = -1, vmax = 1, linewidth = 0.5,
			mask = filtro, annot_kws={"size": 6})
fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} DE DADOS BRUTOS EM 2022.\nMUNICÍPIO DE PORTO ALEGRE, RETROAGINDO ATÉ {_retroceder[2]} DIAS.", weight = "bold", size = "medium")
ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
if _SALVAR == "True":
	caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
	os.makedirs(caminho_correlacao, exist_ok = True)
	plt.savefig(f'{caminho_correlacao}matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[2]}d.pdf', format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
	print(f"""\n{ansi['green']}SALVO COM SUCESSO!\n
	{ansi['cyan']}ENCAMINHAMENTO: {caminho_correlacao}\n
	NOME DO ARQUIVO: matriz_correlacao_{_METODO}_dadosbrutos2022_Porto_Alegre_r{_retroceder[2]}d.pdf{ansi['reset']}\n""")
if _VISUALIZAR == "True":
	print(f"{ansi['green']}Exibindo a Matriz de Correlação de {_METODO.title()} de dados brutos em 2022 do município de Porto Alegre com até {_retroceder[2]} dias retroagidos. {ansi['reset']}")
	plt.show()
#sys.exit()

