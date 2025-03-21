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
###############################

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
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"
caminho_indice = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_top20 = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/correlacoes/top20/"
print(f"{green}CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{cyan}{caminho_dados}{reset}\n\n")

### Renomeação das Variáveis pelos Arquivos
meteoro = "meteoro_porto_alegre.csv"
iam3 = "serie_IAM3_porto_alegre.csv"

### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}")
iam3 = pd.read_csv(f"{caminho_indice}{iam3}")
print(f"\n{green}meteoro:\n{reset}{meteoro}\n")
print(f"\n{green}iam3:\n{reset}{iam3}\n")
sys.exit()

### Pré-Processamento
### Montando Dataset
dataset = biometeoro.copy()
dataset.dropna(inplace = True)
print(dataset)

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

print(dataset)

#sys.exit()

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
sobre = RandomOverSampler(sampling_strategy = "minority")
sub = RandomUnderSampler(sampling_strategy = "majority")
treino_x, treino_y = sobre.fit_resample(treino_x, treino_y)
treino_x, treino_y = sub.fit_resample(treino_x, treino_y)

explicativas = x.columns.tolist() # feature_names = explicativas
treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
treino_x_explicado = treino_x_explicado.to_numpy().astype(int)

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

#sys.exit()

#########################################################FUNÇÕES###############################################################
### Definições
def monta_dataset(cidade):
	dataset = tmin[["Data"]].copy()
	dataset["TMIN"] = tmin[cidade].copy()
	dataset["TMED"] = tmed[cidade].copy()
	dataset["TMAX"] = tmax[cidade].copy()
	dataset = dataset.merge(prec[["Data", cidade]], how = "left", on = "Data").copy()
	dataset = dataset.merge(obitos[["Data", cidade]], how = "left", on = "Data").copy()
	troca_nome = {f"{cidade}_x" : "PREC", f"{cidade}_y" : "obito"}
	dataset = dataset.rename(columns = troca_nome)
	for r in range(5, _RETROAGIR + 1):
		dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
		dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
		dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
		dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
		dataset[f"obitos_r{r}"] = dataset["obito"].shift(-r)
	dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
	dataset.dropna(inplace = True)
	dataset.set_index("Data", inplace = True)
	dataset.columns.name = f"{cidade}"
	return dataset

def treino_teste(dataset, cidade, tamanho_teste = 0.2):
	SEED = np.random.seed(0)
	x = dataset.drop(columns = "obito")
	y = dataset["obito"]
	if x.empty or x.isnull().all().all():
		print(f"'X' está vazio ou contém apenas valores 'NaN! Confira o dataset do município {cidade}!")
		print(f"{cidade} possui um conjunto com erro:\n {x}")
		return None, None, None, None, None
	x = x.dropna()
	if x.empty:
		print(f"'X' continua vazio, mesmo removendo valores 'NaN'! Confira o dataset do município {cidade}!")
		print(f"{cidade} possui um conjunto com erro:\n {x}")
		return None, None, None, None, None
	if y.empty or y.isnull().all().all():
		print(f"'Y' está vazio ou contém apenas valores 'NaN! Confira o dataset do município {cidade}!")
		print(f"{cidade} possui um conjunto com erro:\n {y}")
		return None, None, None, None, None
	y = y.dropna()
	if y.empty:
		print(f"'Y' continua vazio, mesmo removendo valores 'NaN'! Confira o dataset do município {cidade}!")
		print(f"{cidade} possui um conjunto com erro:\n {y}")
		return None, None, None, None, None
	x_array = x.to_numpy()
	x_array = x_array.reshape(x_array.shape[0], -1)
	x_array = x.to_numpy().astype(int)
	y_array = y.to_numpy().astype(int)
	x_array = x_array.reshape(x_array.shape[0], -1)
	treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array,
		                                                random_state = SEED,
		                                                test_size = tamanho_teste)
	explicativas = x.columns.tolist()
	treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
	treino_x_explicado = treino_x_explicado.to_numpy().astype(int)
	return treino_x, teste_x, treino_y, teste_y, treino_x_explicado

def escalona(treino_x, teste_x):
	escalonador = StandardScaler()
	escalonador.fit(treino_x)
	treino_normal_x = escalonador.transform(treino_x)
	teste_normal_x = escalonador.transform(teste_x)
	return treino_normal_x, teste_normal_x

def RF_modela_treina_preve(treino_x, treino_y, teste_x, SEED):
	modelo = RandomForestRegressor(n_estimators = 100, random_state = SEED)
	modelo.fit(treino_x_explicado, treino_y)
	y_previsto = modelo.predict(teste_x)
	previsoes = modeloRF.predict(x)
	previsoes = [int(p) for p in previsoes]
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

def salva_modeloRF(modelo, cidade):
	if not os.path.exists(caminho_modelos):
		os.makedirs(caminho_modelos)
	joblib.dump(modelo, f"{caminho_modelos}RF_obitos_r{_RETROAGIR}_{_cidade}.h5")
	print(f"\nMODELO RANDOM FOREST DE {cidade} SALVO!\n\nCaminho e Nome:\n {caminho_modelos}RF_obitos_r{_RETROAGIR}_{_cidade}.h5")
	print("\n" + "="*80 + "\n")

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

def grafico_previsao(previsao, teste, string_modelo):
	if string_modelo not in ["RF", "NN"]:
		print("!!"*80)
		print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
		print("!!"*80)
		sys.exit()
	# Gráfico de Comparação entre Observação e Previsão dos Modelos
	nome_modelo = "Random Forest" if string_modelo == "RF" else "Rede Neural"
	final = pd.DataFrame()
	final["Data"] = data_dataset["data"]
	final["obito"] = data_dataset["obito"]
	final.reset_index(inplace = True)
	final.drop(columns = "index", inplace = True)
	print(final)
	#sys.exit()
	final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
	final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
	print(final)
	previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
	"""
	lista_previsao = [previsoes[v] for v in range(len(previsoes))]
	final["Previstos"] = lista_previsao
	"""
	previsoes = previsoes[:len(final)]
	final["Previstos"] = previsoes
	final["Data"] = pd.to_datetime(final["Data"])
	print(final)
	print("="*80)
	plt.figure(figsize = (10, 6), layout = "tight", frameon = False)
	sns.lineplot(x = final["Data"], y = final["obito"], # linestyle = "--" linestyle = "-."
		     	color = "darkblue", linewidth = 1, label = "Observado")
	sns.lineplot(x = final["Data"], y = final["Previstos"],
		     	color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
	plt.title(f"MODELO {nome_modelo.upper()} (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade}, RIO GRANDE DO SUL.\n")
	plt.xlabel("Série Histórica (Observação Diária)")
	plt.ylabel("Número de Óbitos Cardiovasculares.")
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
	   'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
	 	'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
	 	'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
	 	'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U', 
	 	'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	_cidade = cidade
	for velho, novo in troca.items():
		_cidade = _cidade.replace(velho, novo)
	if _SALVAR == "True":
		plt.savefig(f'{caminho_resultados}modelo_RF_obitos_{_cidade}.pdf', format = "pdf", dpi = 1200)
		print(f"{ansi['green']}\nSALVANDO:\n{caminho_resultados}modelo_RF_obitos_{_cidade}.pdf{ansi['reset']}")
	if _VISUALIZAR == "True":	
		print(f"{ansi['green']}\nVISUALIZANDO:\n{caminho_resultados}modelo_RF_obitos_{_cidade}.pdf{ansi['reset']}")
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

def metricas_importancias(modeloRF, explicativas):
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
	ax.set_title(f"VARIÁVEIS IMPORTANTES PARA MODELO RANDOM FOREST\nMUNICÍPIO DE {cidade}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel("Impureza Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 60)
	for i, v in enumerate(importancia_impureza[:10].values):
		ax.text(i + 0.01, v, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	if _SALVAR == "True":
		plt.savefig(f'{caminho_resultados}importancias_modelo_RF_obitos_{_cidade}.pdf', format = "pdf", dpi = 1200)
		print(f"{ansi['green']}\nSALVANDO:\n{caminho_resultados}importancias_modelo_RF_obitos_{_cidade}.pdf{ansi['reset']}")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}\nVISUALIZANDO:\n{caminho_resultados}importancias_modelo_RF_obitos_{_cidade}.pdf{ansi['reset']}")
		plt.show()
	#2 Permutações
	n_permuta = 10
	resultado_permuta = permutation_importance(modeloRF, teste_x, teste_y, n_repeats = n_permuta, random_state = SEED, n_jobs = 2)
	importancia_permuta = pd.Series(resultado_permuta.importances_mean, index = explicativas)
	importancia_permuta = importancia_permuta.sort_values(ascending = False)
	std = resultado_permuta.importances_std
	fig, ax = plt.subplots(figsize = (10, 6), layout = "tight", frameon = False)
	importancia_permuta[:10].plot.bar(yerr = std[:10], ax = ax)
	ax.set_title(f"VARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO ({n_permuta})\nMUNICÍPIO DE {cidade}, RIO GRANDE DO SUL.\n")
	ax.set_ylabel("Acurácia Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Óbitos Cardiovasculares")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 60)
	for i, v in enumerate(importancia_permuta[:10].values):
		ax.text(i + 0.01, v, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	if _SALVAR == "True":
		plt.savefig(f'{caminho_resultados}importancias_permuta_RF_obitos_{_cidade}.pdf', format = "pdf", dpi = 1200)
		print(f"{ansi['green']}\nSALVANDO:\n{caminho_resultados}importancias_permuta_RF_obitos_{_cidade}.pdf{ansi['reset']}")
	if _VISUALIZAR == "True":
		print(f"{ansi['green']}\nVISUALIZANDO:\n{caminho_resultados}importancias_permuta_RF_obitos_{_cidade}.pdf{ansi['reset']}")
		plt.show()
	print(f"\nVARIÁVEIS IMPORTANTES:\n{importancia_impureza}\n")
	print(f"\nVARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO:\n{importancia_permuta}")
	return importancias, indices, variaveis_importantes

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
		if _SALVAR == "True":
			plt.savefig(f'{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.pdf', format = "pdf", dpi = 1200)
			print(f"\n{ansi['green']}ARQUIVO SALVO COM SUCESSO\n\n{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.pdf{ansi['reset']}\n")
			with open(f'{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.txt', 'w') as file:
				file.write(relatorio_decisao)
			print(f"\n{ansi['green']}ARQUIVO SALVO COM SUCESSO\n\n{caminho_resultados}arvore_decisao_modelo_RF_{_cidade}.txt{ansi['reset']}\n")
		if _VISUALIZAR == "True":
			print("\n\n{ansi['green']}RELATÓRIO DA ÁRVORE DE DECISÃO\n\n{cidade}\n\n{cidade}{ansi['reset']}\n\n", relatorio_decisao)
			plt.show()
		#print("\n\nCAMINHO DE DECISÃO\n\n", caminho_denso)
		return unica_arvore, relatorio_decisao #amostra, caminho, caminho_denso

def salva_modelo(string_modelo, modeloNN = None):
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
			modeloNN.save(modeloNN, f"{caminho_modelos}NN_obitos_r{_RETROAGIR}_{_cidade}.h5")
	else:
		joblib.dump(modeloRF, f"{caminho_modelos}RF_obitos_r{_RETROAGIR}_{_cidade}.h5")

######################################################RANDOM_FOREST############################################################

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

#########################################################AUTOMATIZANDO###############################################################
if _AUTOMATIZAR == "True":
    for cidade in cidades:
        dataset = monta_dataset(cidade)
        treino_x, teste_x, treino_y, teste_y, treino_x_explicado = treino_teste(dataset, cidade)
        modelo, y_previsto, previsoes = RF_modela_treina_preve(treino_x_explicado, treino_y, teste_x, SEED)
        EQM, RQ_EQM, R_2 = RF_previsao_metricas(dataset, previsoes, 5, teste_y, y_previsto)
        salva_modeloRF(modelo, cidade)
######################################################################################################################################

