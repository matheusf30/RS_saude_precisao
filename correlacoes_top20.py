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
		caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
	case "CLUSTER":
		caminho_dados = "..."
	case "CASA":
		caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/"
		caminho_dados = "/home/mfsouza90/Documents/git_matheusf30/dados_dengue/modelos/"
	case _:
		print("CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteoro_porto_alegre.csv"
iam1 = "serie_IAM1_porto_alegre.csv"
iam2 = "serie_IAM2_porto_alegre.csv"
iam3 = "serie_IAM3_porto_alegre.csv"
iam4 = "serie_IAM4_porto_alegre.csv"

### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}")
iam1 = pd.read_csv(f"{caminho_indice}{iam1}")
iam2 = pd.read_csv(f"{caminho_indice}{iam2}")
iam3 = pd.read_csv(f"{caminho_indice}{iam3}")
iam4 = pd.read_csv(f"{caminho_indice}{iam4}")
print(f"\n{green}Meteoro PoA\n{reset}{meteoro}\n")
print(f"\n{green}IAM1\n{reset}{iam1}\n")
print(f"\n{green}IAM2\n{reset}{iam2}\n")
print(f"\n{green}IAM3\n{reset}{iam3}\n")
print(f"\n{green}IAM4\n{reset}{iam4}\n")
print(f"\n{green}IAM4.columns\n{reset}{iam4.columns}\n")

#Definindo Funções
# Selecionando sazonalidade
def selecionando_periodos(meteoro):
	inverno = meteoro.copy()
	inverno["dia"] = inverno["data"]
	inverno.set_index("dia", inplace = True)
	inverno.index = pd.to_datetime(inverno.index)
	inverno = inverno[(inverno.index.month >= 4) & (inverno.index.month <= 9)]
	inverno.reset_index(inplace = True)
	inverno.drop(columns = "dia", inplace = True)
	print(f"\n{green}PERÍODO SELECIONADO: INVERNO\n{reset}{inverno}\n{inverno.info()}\n")
	verao = meteoro.copy()
	verao["dia"] = verao["data"]
	verao.set_index("dia", inplace = True)
	verao.index = pd.to_datetime(verao.index)
	verao = verao[(verao.index.month == 12) | (verao.index.month <= 2)]
	verao.reset_index(inplace = True)
	verao.drop(columns = "dia", inplace = True)
	print(f"\n{green}PERÍODO SELECIONADO: VERÃO\n{verao}\n{reset}{verao.info()}\n")
	return inverno, verao
	
def concatenando_indices_inner(csv):
	csv_in_1 = csv.merge(iam1, on = "data", how = "inner")
	csv_in_2 = csv.merge(iam2, on = "data", how = "inner")
	csv_in_3 = csv.merge(iam3, on = "data", how = "inner")
	csv_in_4 = csv.merge(iam4, on = "data", how = "inner")
	print(f"\n{green}IAM1 INNER\n{reset}{csv_in_1}\n")
	print(f"\n{green}IAM2 INNER\n{reset}{csv_in_2}\n")
	print(f"\n{green}IAM3 INNER\n{reset}{csv_in_3}\n")
	print(f"\n{green}IAM4 INNER\n{reset}{csv_in_4}\n")
	print(f"\n{green}IAM4 INNER (COLUMNS)\n{reset}{csv_in_4.columns}\n")
	return csv_in_1, csv_in_2, csv_in_3, csv_in_4

def concatenando_indices_outer(csv):
	csv_out_1 = csv.merge(iam1, on = "data", how = "outer").fillna(0)
	csv_out_2 = csv.merge(iam2, on = "data", how = "outer").fillna(0)
	csv_out_3 = csv.merge(iam3, on = "data", how = "outer").fillna(0)
	csv_out_4 = csv.merge(iam4, on = "data", how = "outer").fillna(0)
	print(f"\n{green}IAM1 OUTER\n{reset}{csv_out_1}\n")
	print(f"\n{green}IAM2 OUTER\n{reset}{csv_out_2}\n")
	print(f"\n{green}IAM3 OUTER\n{reset}{csv_out_3}\n")
	print(f"\n{green}IAM4 OUTER\n{reset}{csv_out_4}\n")
	print(f"\n{green}IAM4 OUTER (COLUMNS)\n{reset}{csv_out_4.columns}\n")
	return csv_out_1, csv_out_2, csv_out_3, csv_out_4

def correlacao_sem_retroacao(lista_arquivos, str_arq):
	lista_METODO = ["pearson", "spearman"]
	IAMs = ["IAM1", "IAM2", "IAM3", "IAM4"]
	colunas_retirar = ["total_top5", "porcent_top5", "total_top10", "porcent_top10",
						"total_top15", "porcent_top15", "total_top20", "porcent_top20"]
	for idx, arquivo in enumerate(lista_arquivos):
		arquivo.set_index("data", inplace = True)
		arquivo.drop(columns = colunas_retirar, inplace = True)
		arquivo.dropna(inplace = True)
		for _METODO in lista_METODO:
			IAM = IAMs[idx]
			nome_arquivo = f"matriz_correlacao_{_METODO}_{IAM}_{str_arq}_top20_Porto_Alegre.pdf"
			correlacao_dataset = arquivo.corr(method = f"{_METODO}")
			print(f"\n{green}{nome_arquivo}\n{reset}{correlacao_dataset}\n")
			fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
			filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
			sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
			fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} ENTRE DADOS METEOROLÓGICOS E PRINCIPAIS ÓBITOS CARDIOVASCULARES.\nMUNICÍPIO DE PORTO ALEGRE, ÍNDICE DE ALTA MORTALIDADE ({IAM}) .",
						weight = "bold", size = "medium")
			ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
			ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
			if _SALVAR == "True":
				caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
				os.makedirs(caminho_correlacao, exist_ok = True)
				plt.savefig(f"{caminho_correlacao}{nome_arquivo}", format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
				print(f"""\n{green}SALVO COM SUCESSO!\n
				{cyan}ENCAMINHAMENTO: {caminho_correlacao}\n
				NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
			if _VISUALIZAR == "True":
				print(f"{green}Exibindo a Matriz de Correlação de {_METODO.title()}. Município de Porto Alegre, {IAM}{reset}")
				plt.show()

def correlacao_retroagindo(lista_arquivos, str_arq):
	retroagir = [1, 2, 3, 4, 5, 6, 7]
	lista_METODO = ["pearson", "spearman"]
	IAMs = ["IAM1", "IAM2", "IAM3", "IAM4"]
	colunas_retirar = ["total_top5", "porcent_top5", "total_top10", "porcent_top10",
						"total_top15", "porcent_top15", "total_top20", "porcent_top20"]
	colunas_r = ["tmin", "temp", "tmax", "amplitude_t",
			"urmin", "umidade", "urmax", "prec",
			"pressao", "ventodir", "ventovel"]
	for idx, arquivox in enumerate(lista_arquivos):
		arquivox.set_index("data", inplace = True)
		arquivox.drop(columns = colunas_retirar, inplace = True)
		arquivox.dropna(inplace = True)
		for _METODO in lista_METODO:
			for r in retroagir:
				IAM = IAMs[idx]
				arquivo = arquivox.copy()
				nome_arquivo = f"matriz_correlacao_{_METODO}_{IAM}_{str_arq}_r{r}_top20_Porto_Alegre.pdf"
				for c in colunas_r:
					arquivo[f"{c}_r{r}"] = arquivo[c].shift(-r)
				arquivo.dropna(inplace = True)
				arquivo.drop(columns = colunas_r, inplace = True)
				print(f"\n{green}{nome_arquivo}\n{reset}{arquivo}\n")
				correlacao_dataset = arquivo.corr(method = f"{_METODO}")
				print(f"\n{green}{nome_arquivo}\n{reset}{correlacao_dataset}\n")
				fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
				filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
				sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
				fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} ENTRE DADOS METEOROLÓGICOS E PRINCIPAIS ÓBITOS CARDIOVASCULARES.\nMUNICÍPIO DE PORTO ALEGRE, ÍNDICE DE ALTA MORTALIDADE ({IAM}) .",
							weight = "bold", size = "medium")
				ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
				ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
				if _SALVAR == "True":
					caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
					os.makedirs(caminho_correlacao, exist_ok = True)
					plt.savefig(f"{caminho_correlacao}{nome_arquivo}", format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
					print(f"""\n{green}SALVO COM SUCESSO!\n
					{cyan}ENCAMINHAMENTO: {caminho_correlacao}\n
					NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
				if _VISUALIZAR == "True":
					print(f"{green}Exibindo a Matriz de Correlação de {_METODO.title()}. Município de Porto Alegre, {IAM}{reset}")
					plt.show()



# Executando Funções
inverno, verao = selecionando_periodos(meteoro)
# Correlações sem retroação
meteoro_in_1, meteoro_in_2, meteoro_in_3, meteoro_in_4 = concatenando_indices_inner(meteoro)
inverno_in_1, inverno_in_2, inverno_in_3, inverno_in_4 = concatenando_indices_inner(inverno)
verao_in_1, verao_in_2, verao_in_3, verao_in_4 = concatenando_indices_inner(verao)
lista_in_total = [meteoro_in_1, meteoro_in_2, meteoro_in_3, meteoro_in_4]
lista_in_inverno = [inverno_in_1, inverno_in_2, inverno_in_3, inverno_in_4]
lista_in_verao = [verao_in_1, verao_in_2, verao_in_3, verao_in_4]
correlacao_sem_retroacao(lista_in_total, "total")
correlacao_sem_retroacao(lista_in_inverno, "inverno")
correlacao_sem_retroacao(lista_in_verao, "verao")
# Correlações Retroagindo até 7 Dias
meteoro_out_1, meteoro_out_2, meteoro_out_3, meteoro_out_4 = concatenando_indices_outer(meteoro)
inverno_out_1, inverno_out_2, inverno_out_3, inverno_out_4 = concatenando_indices_outer(inverno)
verao_out_1, verao_out_2, verao_out_3, verao_out_4 = concatenando_indices_outer(verao)
lista_out_total = [meteoro_out_1, meteoro_out_2, meteoro_out_3, meteoro_out_4]
lista_out_inverno = [inverno_out_1, inverno_out_2, inverno_out_3, inverno_out_4]
lista_out_verao = [verao_out_1, verao_out_2, verao_out_3, verao_out_4]
correlacao_retroagindo(lista_out_total, "total")
correlacao_retroagindo(lista_out_inverno, "inverno")
correlacao_retroagindo(lista_out_verao, "verao")
sys.exit()

### Concatenando CIDs e Meteorologia
# Apenas os dias filtrados (Série Total)
meteoro1_in = meteoro.merge(iam1, on = "data", how = "inner")
meteoro2_in = meteoro.merge(iam2, on = "data", how = "inner")
meteoro3_in = meteoro.merge(iam3, on = "data", how = "inner")
meteoro4_in = meteoro.merge(iam4, on = "data", how = "inner")
print(f"\n{green}IAM1 INNER\n{reset}{meteoro1_in}\n")
print(f"\n{green}IAM2 INNER\n{reset}{meteoro2_in}\n")
print(f"\n{green}IAM3 INNER\n{reset}{meteoro3_in}\n")
print(f"\n{green}IAM4 INNER\n{reset}{meteoro4_in}\n")
print(f"\n{green}IAM4 INNER (COLUMNS)\n{reset}{meteoro4_in.columns}\n")
# Apenas os dias filtrados (Período Frio)
inverno1_in = inverno.merge(iam1, on = "data", how = "inner")
inverno2_in = inverno.merge(iam2, on = "data", how = "inner")
inverno3_in = inverno.merge(iam3, on = "data", how = "inner")
inverno4_in = inverno.merge(iam4, on = "data", how = "inner")
print(f"\n{green}IAM1 INNER\n{reset}{inverno1_in}\n")
print(f"\n{green}IAM2 INNER\n{reset}{inverno2_in}\n")
print(f"\n{green}IAM3 INNER\n{reset}{inverno3_in}\n")
print(f"\n{green}IAM4 INNER\n{reset}{inverno4_in}\n")
print(f"\n{green}IAM4 INNER (COLUMNS)\n{reset}{inverno4_in.columns}\n")
# Apenas os dias filtrados (Período Quente)
verao1_in = verao.merge(iam1, on = "data", how = "inner")
verao2_in = verao.merge(iam2, on = "data", how = "inner")
verao3_in = verao.merge(iam3, on = "data", how = "inner")
verao4_in = verao.merge(iam4, on = "data", how = "inner")
print(f"\n{green}IAM1 INNER\n{reset}{verao1_in}\n")
print(f"\n{green}IAM2 INNER\n{reset}{verao2_in}\n")
print(f"\n{green}IAM3 INNER\n{reset}{verao3_in}\n")
print(f"\n{green}IAM4 INNER\n{reset}{verao4_in}\n")
print(f"\n{green}IAM4 INNER (COLUMNS)\n{reset}{verao4_in.columns}\n")
# Iniciando laços de Correlações
lista_METODO = ["pearson", "spearman"]#, "pearson", "spearman"
colunas_r = ["tmin", "temp", "tmax", "amplitude_t",
			"urmin", "umidade", "urmax", "prec",
			"pressao", "ventodir", "ventovel"]
lista_arquivos = [meteoro1_in, meteoro2_in, meteoro3_in, meteoro4_in]
IAMs = ["IAM1", "IAM2", "IAM3", "IAM4"]
colunas_retirar = ["total_top5", "porcent_top5", "total_top10", "porcent_top10",
				"total_top15", "porcent_top15", "total_top20", "porcent_top20"]

for idx, arquivo in enumerate(lista_arquivos):
	arquivo.set_index("data", inplace = True)
	arquivo.drop(columns = colunas_retirar, inplace = True)
	arquivo.dropna(inplace = True)
	for _METODO in lista_METODO:
		IAM = IAMs[idx]
		nome_arquivo = f"matriz_correlacao_{_METODO}_{IAM}_top20_Porto_Alegre.pdf"
		correlacao_dataset = arquivo.corr(method = f"{_METODO}")
		print(f"\n{green}{nome_arquivo}\n{reset}{correlacao_dataset}\n")
		fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
		filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
		sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
		fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} ENTRE DADOS METEOROLÓGICOS E PRINCIPAIS ÓBITOS CARDIOVASCULARES.\nMUNICÍPIO DE PORTO ALEGRE, ÍNDICE DE ALTA MORTALIDADE ({IAM}) .",
					weight = "bold", size = "medium")
		ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
		ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
		if _SALVAR == "True":
			caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
			os.makedirs(caminho_correlacao, exist_ok = True)
			plt.savefig(f"{caminho_correlacao}{nome_arquivo}", format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)

			print(f"""\n{green}SALVO COM SUCESSO!\n
			{cyan}ENCAMINHAMENTO: {caminho_correlacao}\n
			NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
		if _VISUALIZAR == "True":
			print(f"{green}Exibindo a Matriz de Correlação de {_METODO.title()}. Município de Porto Alegre, {IAM}{reset}")
			plt.show()

# Realizando correlações

sys.exit()

retroagir = [1, 2, 3, 4, 5, 6, 7]
for idx, arquivox in enumerate(lista_arquivos):
	arquivox.set_index("data", inplace = True)
	arquivox.drop(columns = colunas_retirar, inplace = True)
	arquivox.dropna(inplace = True)
	for _METODO in lista_METODO:
		for r in retroagir:
			arquivo = arquivox.copy()
			IAM = IAMs[idx]
			nome_arquivo = f"matriz_correlacao_{_METODO}_{IAM}_top20_r{r}_Porto_Alegre.pdf"
			arquivo[f"tmin_r{r}"] = arquivo["tmin"].shift(-r)
			arquivo[f"temp_r{r}"] = arquivo["temp"].shift(-r)
			arquivo[f"tmax_r{r}"] = arquivo["tmax"].shift(-r)
			arquivo[f"amplitude_t_r{r}"] = arquivo["amplitude_t"].shift(-r)
			arquivo[f"urmin_r{r}"] = arquivo["urmin"].shift(-r)
			arquivo[f"umidade_r{r}"] = arquivo["umidade"].shift(-r)
			arquivo[f"urmax_r{r}"] = arquivo["urmax"].shift(-r)
			arquivo[f"prec_r{r}"] = arquivo["prec"].shift(-r)
			arquivo[f"pressao_r{r}"] = arquivo["pressao"].shift(-r)		
			arquivo[f"ventodir_r{r}"] = arquivo["ventodir"].shift(-r)
			arquivo[f"ventovel_r{r}"] = arquivo["ventovel"].shift(-r)
			arquivo.dropna(inplace = True)
			arquivo.drop(columns = colunas_r, inplace = True)
			print(f"\n{green}{nome_arquivo}\n{reset}{arquivo}\n")
			correlacao_dataset = arquivo.corr(method = f"{_METODO}")
			print(f"\n{green}{nome_arquivo}\n{reset}{correlacao_dataset}\n")
			fig, ax = plt.subplots(figsize = (18, 8), layout = "constrained", frameon = False)
			filtro = np.triu(np.ones_like(correlacao_dataset, dtype = bool), k = 1)
			sns.heatmap(correlacao_dataset, annot = True, cmap = "Spectral", vmin = -1, vmax = 1, linewidth = 0.5, mask = filtro)
			fig.suptitle(f"MATRIZ DE CORRELAÇÃO DE {_METODO.upper()} ENTRE DADOS METEOROLÓGICOS E PRINCIPAIS ÓBITOS CARDIOVASCULARES.\nMUNICÍPIO DE PORTO ALEGRE, ÍNDICE DE ALTA MORTALIDADE ({IAM}), RETROAGINDO ({r}d).",
						weight = "bold", size = "medium")
			ax.set_yticklabels(ax.get_yticklabels(), rotation = "horizontal")
			ax.set_xticklabels(ax.get_xticklabels(), rotation = 75)
			if _SALVAR == "True":
				caminho_correlacao = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/"
				os.makedirs(caminho_correlacao, exist_ok = True)
				plt.savefig(f"{caminho_correlacao}{nome_arquivo}", format = "pdf", dpi = 1200,  bbox_inches = "tight", pad_inches = 0.0)
				print(f"""\n{green}SALVO COM SUCESSO!\n
				{cyan}ENCAMINHAMENTO: {caminho_correlacao}\n
				NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
			if _VISUALIZAR == "True":
				print(f"{green}Exibindo a Matriz de Correlação de {_METODO.title()}. Município de Porto Alegre, {IAM}{reset}")
				plt.show()
print(f"\n{green}IAM DATAS FILTRADAS\n{reset}{arquivo}\n")

sys.exit()
#Toda a série histórica
meteoro1_out = meteoro.merge(iam1, on = "data", how = "outer").fillna(0)
meteoro2_out = meteoro.merge(iam2, on = "data", how = "outer").fillna(0)
meteoro3_out = meteoro.merge(iam3, on = "data", how = "outer").fillna(0)
meteoro4_out = meteoro.merge(iam4, on = "data", how = "outer").fillna(0)
print(f"\n{green}IAM1 OUTER\n{reset}{meteoro1_out}\n")
print(f"\n{green}IAM2 OUTER\n{reset}{meteoro2_out}\n")
print(f"\n{green}IAM3 OUTER\n{reset}{meteoro3_out}\n")
print(f"\n{green}IAM4 OUTER\n{reset}{meteoro4_out}\n")
print(f"\n{green}IAM4 OUTER (COLUMNS)\n{reset}{meteoro4_out.columns}\n")
<<<<<<< HEAD
=======






>>>>>>> f19744377ea22ce95d1c8a63671575a73cad1f66
