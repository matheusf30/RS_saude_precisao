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
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.inspection import permutation_importance
matplotlib.use("Agg")
import shap
print(f"shap.__version__ = {shap.__version__}") #0.46.0
# Modelos e Visualizações
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import export_graphviz, export_text, plot_tree

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
_VISUALIZAR = sys.argv[1]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[2]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################
##################################################################
        
_RETROAGIR = 7 # Dias
_HORIZONTE = 0 # Tempo de Previsão

cidade = "Porto Alegre"
cidades = ["Porto Alegre"]
_CIDADE = cidade.upper()
_cidade = _CIDADE.replace(" ", "_")
print(_cidade)
#sys.exit()

#########################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_indices = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
caminho_shap = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/SHAP/"
caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/modelos/"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
clima = "climatologia.csv"
anomalia = "anomalia.csv"
anomalia_estacionaria = "anomalia_estacionaria.csv"
bio = "obito_cardiovasc_total_poa_96-22.csv"
biometeoro = "biometeoro_PORTO_ALEGRE.csv"


### Abrindo Arquivo e Pré-processamento
clima = pd.read_csv(f"{caminho_dados}{clima}", low_memory = False)
anomalia = pd.read_csv(f"{caminho_dados}{anomalia}", low_memory = False)
anomalia_estacionaria = pd.read_csv(f"{caminho_dados}{anomalia_estacionaria}", low_memory = False)
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
biometeoro = pd.read_csv(f"{caminho_dados}{biometeoro}", low_memory = False)

print(f"\n{green}CLIMATOLOGIA:\n{reset}{clima}\n")
print(f"\n{green}ANOMALIAS:\n{reset}{anomalia}\n")
print(f"\n{green}ANOMALIAS ESTACIONÁRIAS\n{reset}{anomalia_estacionaria}\n")
print(f"\n{green}DADOS DE SAÚDE:\n{reset}{bio}\n")
print(f"\n{green}DADOS BIOMETEOROLÓGICOS:\n{reset}{biometeoro}\n")
print(f"\n{green}COLUNAS DE DADOS BIOMETEOROLÓGICOS:\n{reset}{biometeoro.columns}\n")
#sys.exit()

### Pré-montando Datasets

dataset = biometeoro.copy()
dataset.dropna(inplace = True)
dataset_original = dataset.copy()

colunas_retroagir = ["heat_index", "wind_chill", "tmin", "temp", "tmax",
					"amplitude_t", "urmin", "umidade", "urmax", "prec", "pressao",
					"ventodir", "ventovel"]

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

def treino_teste(dataset, cidade, tamanho_teste = 0.2):
	SEED = np.random.seed(0)
	x = dataset.drop(columns = ["obito"])
	y = dataset["obito"]
	print(f"\n{green}x:\n{reset}{x}\n")
	print(f"\n{green}y:\n{reset}{y}\n")	
	x_array = x.to_numpy().astype(int)
	y_array = y.to_numpy().astype(float)
	treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array, 
		                                                random_state = SEED,
		                                                test_size = tamanho_teste)
	explicativas = x.columns.tolist()
	treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
	df_treino_x_explicado = treino_x_explicado.copy()
	treino_x_explicado = treino_x_explicado.to_numpy().astype(int)
	print(f"\n{green}explicativas:\n{reset}{explicativas}\n")
	print(f"\n{green}treino_x_explicado:\n{reset}{treino_x_explicado}\n")
	return x_array, y_array, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, df_treino_x_explicado, explicativas, SEED

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

def RF_previsao_metricas(dataset, previsoes, n, teste_y, y_previsto):
	nome_modelo = "Random Forest"
	print("="*80)
	print(f"\n{nome_modelo.upper()} - {cidade}\n")
	lista_op = [f"obitos: {dataset['obito'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
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

def grafico_previsao(dataset, previsao):
	nome_modelo = "Random Forest"
	final = pd.DataFrame()
	dataset.reset_index(inplace = True)
	final["Data"] = dataset["data"]
	final["obito"] = dataset["obito"]
	nome_arquivo = "totais"
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
	plt.title(f"MODELO {nome_modelo.upper()}: OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
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
		plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "pdf", dpi = 1200)
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

def metricas_importancias(modeloRF, explicativas, teste_x, teste_y):
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
	nome_arquivo = "totais"
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
	nome_arquivo = f"totais"
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

def metrica_shap(modelo, treino_x, teste_x):
	expl_shap = shap.Explainer(modelo, treino_x)
	valor_shap = expl_shap(teste_x)
	plt.figure(figsize = (10, 6)).set_layout_engine(None)
	nome_arquivo = "totais"
	ax = plt.gca()
	ax.set_title(f"SHAP (SHapley Additive exPlanations) PARA MODELO RANDOM FOREST\n({nome_arquivo}), MUNICÍPIO DE {cidade.upper()}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel(f"Valor SHAP ({nome_arquivo})")
	ax.set_xlabel(f"Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares ({nome_arquivo})")
	ax.set_facecolor("honeydew")
	plt.rcParams.update({"figure.autolayout" : False})
	shap.summary_plot(valor_shap, teste_x, feature_names = list(treino_x.columns))#, legacy_colorbar = True)
	nome_arquivo_pdf = f"importancias_SHAP_modelo_RF_{nome_arquivo}_{_cidade}.pdf"
	nome_arquivo_csv = f"importancias_SHAP_modelo_RF_{nome_arquivo}_{_cidade}.csv"
	if _SALVAR == True:
		plt.savefig(f"{caminho_resultados}{nome_arquivo_pdf}", format = "pdf", dpi = 1200)
		print(f"{green}\nSALVANDO:\n{caminho_resultados}{nome_arquivo_pdf}{reset}")
		nome_treino_x = treino_x.T
		nome_treino_x.reset_index(inplace = True)
		nome_treino_x = nome_treino_x.iloc[:,:1]
		nome_treino_x = nome_treino_x.rename(columns = {"index" : "variavel"})
		nome_treino_x["indice"] = [f"Feature {i}" for i in range(0, len(nome_treino_x))]
		nome_treino_x = nome_treino_x[["indice", "variavel"]]
		nome_treino_x.to_csv(f"{caminho_resultados}{nome_arquivo_csv}", index = False)
		print(f"{green}\nSALVANDO:\n{caminho_resultados}{nome_arquivo_csv}{reset}")
	if _VISUALIZAR == True:
		print(f"{green}\nVISUALIZANDO:\n{caminho_resultados}{nome_arquivo}{reset}")
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

def salva_modeloRF(modelo, cidade):
	if not os.path.exists(caminho_modelos):
		os.makedirs(caminho_modelos)
	nome_arquivo = f"RF_obitos_r{_RETROAGIR}_{_cidade}.h5"
	joblib.dump(modelo, f"{caminho_modelos}{nome_arquivo}")
	print(f"\n{green}MODELO RANDOM FOREST DE {cidade} SALVO!\n{reset}")
	print(f"\n{green}Caminho e Nome:\n{bold} {caminho_modelos}{nome_arquivo}\n{reset}")
	print("\n" + f"{green}={cyan}={reset}"*80 + "\n")

######################################################RANDOM_FOREST############################################################

caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/modelagem/"
if not os.path.exists(caminho_resultados):
	os.makedirs(caminho_resultados)
#caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/indice_amplitude/"
caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/modelos/"
if not os.path.exists(caminho_modelos):
	os.makedirs(caminho_modelos)		
dataset = dataset_original.copy()
dataset_montado = monta_dataset(dataset)
x, y, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, df, explicativas, SEED = treino_teste(dataset_montado, cidade)
modelo, y_previsto, previsoes = RF_modela_treina_preve(x, treino_x_explicado, treino_y, teste_x, SEED)
EQM, RQ_EQM, R_2 = RF_previsao_metricas(dataset, previsoes, 5, teste_y, y_previsto)
grafico_previsao(biometeoro, previsoes)
metricas_importancias(modelo, explicativas, teste_x, teste_y)
metrica_shap(modelo, df, teste_x)
salva_modeloRF(modelo, _cidade)

######################################################################################################################################

