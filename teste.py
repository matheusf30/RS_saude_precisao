### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
#import datetime
# Suporte
import os
import sys
import joblib
# Pré-Processamento e Validações
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score#, RocCurveDisplay
from sklearn.inspection import permutation_importance
# Modelos e Visualizações
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.tree import export_graphviz, export_text, plot_tree
#from sklearn.utils.graph import single_source_shortest_path_lenght as short_path

### Condições para Variar #################################################
_LOCAL = "IFSC" # OPÇÕES>>> "GH" "CASA" "IFSC"


_RETROAGIR = 7 # Dias
_HORIZONTE = 0 # Tempo de Previsão
_JANELA_MM = 0 # Média Móvel
_K = 0 # constante para formar MM

cidade = "Porto Alegre"

_AUTOMATIZA = False


z = 6
limite = "out2023"
fim = "nov2023"

z = 19
limite = "jul2023"
fim = "ago2023"

z = 32
limite = "abr2023"
fim = "mai2023"

z = 50
limite = "dez2022"
fim = "jan2023"
"""
"""
obs = f"(Treino até {limite}; Teste após {fim})"
#########################################################################

### Encaminhamento aos Diretórios
_LOCAL = "SIFAPSC" # OPÇÕES>>> "GH" "CASA" "IFSC"
if _LOCAL == "GH": # _ = Variável Privada
	caminho_dados = "https://raw.githubusercontent.com/matheusf30/RS_saude_precisao/main/dados/"
	#caminho_modelos = "https://github.com/matheusf30/dados_dengue/tree/main/modelos/"
elif _LOCAL == "SIFAPSC":
	caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
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
bio = "sinan_total_poa_96-22.csv"
"""
prec = "prec_semana_ate_2023.csv"
tmin = "tmin_semana_ate_2023.csv"
tmed = "tmed_semana_ate_2023.csv"
tmax = "tmax_semana_ate_2023.csv"
"""
### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", skiprows = 10, sep = ";", low_memory = False)
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
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
colunas_objt = meteoro.select_dtypes(include='object').columns
meteoro = meteoro.replace("," , ".")
meteoro[colunas_objt] = meteoro[colunas_objt].apply(lambda x: x.str.replace(",", "."))
meteoro["prec"] = pd.to_numeric(meteoro["prec"], errors = "coerce")
meteoro["pressao"] = pd.to_numeric(meteoro["pressao"], errors = "coerce")
meteoro["temp"] = pd.to_numeric(meteoro["temp"], errors = "coerce")
meteoro["ventovel"] = pd.to_numeric(meteoro["ventovel"], errors = "coerce")
prec = meteoro.groupby(by = ["data"])["prec"].sum()
meteoro = meteoro.groupby(by = "data")[["pressao", "temp", "umidade", "ventodir", "ventovel"]].mean().round(2)
meteoro = meteoro.merge(prec, on = "data", how = "right")

# BIOMETEORO
meteoro.reset_index(inplace = True)
meteoro["data"] = pd.to_datetime(meteoro["data"])
total = total.to_frame(name = "obito")
total.reset_index(inplace = True)
biometeoro = meteoro.merge(total, on = "data", how = "inner")
biometeoro = biometeoro[["data", "obito", "temp", "umidade", "prec", "pressao", "ventodir",  "ventovel"]]


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

# Visualização Prévia
"""
plt.figure(figsize = (18, 6), layout = "constrained", frameon = False)
ax1 = plt.gca()
sns.lineplot(x = biometeoro["data"], y = biometeoro["obito"],
             color = "black", alpha = 0.7, linewidth = 1, label = "Óbitos", ax = ax1)
ax2 = ax1.twinx()
sns.lineplot(x = biometeoro["data"], y = biometeoro["temp"],
             color = "red", alpha = 0.7, linewidth = 1, label = "Temperatura", ax = ax2)
ax1.legend(loc = "upper left")
#sns.lineplot(x = biometeoro["data"], y = biometeoro["prec"],
#             color = "darkblue", alpha = 0.7, linewidth = 3, label = "Precipitação")
plt.show()
"""

### Montando Dataset
dataset = biometeoro.copy()
dataset.dropna(inplace = True)
print(dataset)

for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
    dataset[f"temp_r{r}"] = dataset["temp"].shift(-r)
    dataset[f"umidade_r{r}"] = dataset["umidade"].shift(-r)
    dataset[f"prec_r{r}"] = dataset["prec"].shift(-r)
    dataset[f"pressao_r{r}"] = dataset["pressao"].shift(-r)
    dataset[f"ventodir_r{r}"] = dataset["ventodir"].shift(-r)
    dataset[f"ventovel_r{r}"] = dataset["ventovel"].shift(-r)
#dataset.drop(columns = ["temp", "umidade", "prec", "pressao", "ventodir", "ventovel"], inplace = True)
dataset.dropna(inplace = True)
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
x_ate_limite = x.iloc[:-z]
y_ate_limite = y.iloc[:-z]
xlimite = x.iloc[-z:]
ylimite = y.iloc[-z:]
treino_x = x_ate_limite.copy()
teste_x = xlimite.copy()
treino_y = y_ate_limite.copy()
teste_y = ylimite.copy()

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
x = dataset.drop(columns = "FOCOS")
y = dataset["FOCOS"]
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
#print("="*80)
#print(f"X no formato numpy.ndarray: {x_array}.")
print("="*80)
print(f"Treinando com {len(treino_x)} elementos e testando com {len(teste_x)} elementos.") # Tamanho é igual para dados normalizados
print(f"Formato dos dados (X) nas divisões treino: {treino_x.shape} e teste: {teste_x.shape}.")
print(f"Formato dos dados (Y) nas divisões treino: {treino_y.shape} e teste: {teste_y.shape}.")
print("="*80)

sys.exit()

#########################################################FUNÇÕES###############################################################
### Definições
def monta_dataset(cidade):
    dataset = tmin[["Semana"]].copy()
    dataset["TMIN"] = tmin[cidade].copy()
    dataset["TMED"] = tmed[cidade].copy()
    dataset["TMAX"] = tmax[cidade].copy()
    dataset = dataset.merge(prec[["Semana", cidade]], how = "left", on = "Semana").copy()
    dataset = dataset.merge(focos[["Semana", cidade]], how = "left", on = "Semana").copy()
    troca_nome = {f"{cidade}_x" : "PREC", f"{cidade}_y" : "FOCOS"}
    dataset = dataset.rename(columns = troca_nome)
    for r in range(5, _RETROAGIR + 1):
        dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
        dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
        dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
        dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
        dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
    dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
    dataset.dropna(inplace = True)
    dataset.set_index("Semana", inplace = True)
    dataset.columns.name = f"{cidade}"
    return dataset

def treino_teste(dataset, cidade):
    SEED = np.random.seed(0)
    x = dataset.drop(columns = "FOCOS")
    y = dataset["FOCOS"]
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
                                                        test_size = 0.2)
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
    lista_op = [f"Focos: {dataset['FOCOS'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
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
    troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
    for velho, novo in troca.items():
        cidade = cidade.replace(velho, novo)
    if not os.path.exists(caminho_modelos):
        os.makedirs(caminho_modelos)
    joblib.dump(modelo, f"{caminho_modelos}RF_focos_r{_RETROAGIR}_{cidade}.h5")
    print(f"\nMODELO RANDOM FOREST DE {cidade} SALVO!\n\nCaminho e Nome:\n {caminho_modelos}RF_focos_r{_RETROAGIR}_{cidade}.h5")
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
    lista_op = [f"Focos: {dataset['FOCOS'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
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
    final["Semana"] = focos["Semana"]
    final["Focos"] = focos[cidade]
    final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
    final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
    previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
    """
    lista_previsao = [previsoes[v] for v in range(len(previsoes))]
    final["Previstos"] = lista_previsao
    """
    previsoes = previsoes[:len(final)]
    final["Previstos"] = previsoes
    final["Semana"] = pd.to_datetime(final["Semana"])
    print(final)
    print("="*80)
    plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
    sns.lineplot(x = final["Semana"], y = final["Focos"], # linestyle = "--" linestyle = "-."
                 color = "darkblue", linewidth = 1, label = "Observado")
    sns.lineplot(x = final["Semana"], y = final["Previstos"],
                 color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
    plt.title(f"MODELO {nome_modelo.upper()} (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade}, SANTA CATARINA.\n{obs}")
    plt.xlabel("Semanas Epidemiológicas na Série de Anos")
    plt.ylabel("Número de Focos de _Aedes_ sp.")
    troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
           'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U', 
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
    _cidade = cidade
    for velho, novo in troca.items():
        _cidade = _cidade.replace(velho, novo)
    #plt.savefig(f'{caminho_resultados}verificatualizacao_modelo_RF_focos_{_cidade}_{limite}-{fim}.pdf', format = "pdf", dpi = 1200)
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
	fig, ax = plt.subplots(figsize = (10, 6), layout = "constrained", frameon = False)
	importancia_impureza = importancia_impureza.sort_values(ascending = False)
	importancia_impureza.plot.bar(yerr = std, ax = ax)
	ax.set_title(f"VARIÁVEIS IMPORTANTES PARA MODELO RANDOM FOREST\nMUNICÍPIO DE {cidade}, SANTA CATARINA.\n{obs}")
	ax.set_ylabel("Impureza Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Focos de _Aedes_ sp.")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = "horizontal")
	for i, v in enumerate(importancia_impureza.values):
		ax.text(i, v + 0.01, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	plt.show()
	#2 Permutações
	n_permuta = 10
	resultado_permuta = permutation_importance(modeloRF, teste_x, teste_y, n_repeats = n_permuta, random_state = SEED, n_jobs = 2)
	importancia_permuta = pd.Series(resultado_permuta.importances_mean, index = explicativas)
	importancia_permuta = importancia_permuta.sort_values(ascending = False)
	fig, ax = plt.subplots(figsize = (10, 6), layout = "constrained", frameon = False)
	importancia_permuta.plot.bar(yerr = resultado_permuta.importances_std, ax = ax)
	ax.set_title(f"VARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO ({n_permuta})\nMUNICÍPIO DE {cidade}, SANTA CATARINA.\n{obs}")
	ax.set_ylabel("Acurácia Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Focos de _Aedes_ sp.")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = "horizontal")
	for i, v in enumerate(importancia_permuta.values):
		ax.text(i, v + 0.01, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	plt.show()
	print(f"\nVARIÁVEIS IMPORTANTES:\n{importancia_impureza}\n")
	print(f"\nVARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO:\n{importancia_permuta}")
	return importancias, indices, variaveis_importantes

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
            modeloNN.save(modeloNN, f"{caminho_modelos}NN_focos_r{_RETROAGIR}_{cidade}.h5")
    else:
        joblib.dump(modeloRF, f"{caminho_modelos}RF_focos_r{_RETROAGIR}_{cidade}.h5")

######################################################RANDOM_FOREST############################################################

### Instanciando e Treinando Modelo Regressor Random Forest
modeloRF = RandomForestRegressor(n_estimators = 100, random_state = SEED) #n_estimators = número de árvores
modeloRF.fit(treino_x_explicado, treino_y)

### Testando e Avaliando
y_previstoRF = modeloRF.predict(teste_x)
EQM_RF = mean_squared_error(teste_y, y_previstoRF)
RQ_EQM_RF = np.sqrt(EQM_RF)
R_2 = r2_score(teste_y, y_previstoRF).round(2) 

### Testando e Validando Modelo
testesRF = modeloRF.predict(teste_x)
previsoesRF = modeloRF.predict(x)
previsoesRF = [int(p) for p in previsoesRF]

### Exibindo Informações, Gráficos e Métricas
lista_previsao(previsoesRF, 5, "RF")
grafico_previsao(previsoesRF, testesRF, "RF")
metricas("RF")

importancias, indices, variaveis_importantes =  metricas_importancias(modeloRF, explicativas)
#########################################################AUTOMATIZANDO###############################################################
if _AUTOMATIZA == True:
    for cidade in cidades:
        dataset = monta_dataset(cidade)
        treino_x, teste_x, treino_y, teste_y, treino_x_explicado = treino_teste(dataset, cidade)
        modelo, y_previsto, previsoes = RF_modela_treina_preve(treino_x_explicado, treino_y, teste_x, SEED)
        EQM, RQ_EQM, R_2 = RF_previsao_metricas(dataset, previsoes, 5, teste_y, y_previsto)
        salva_modeloRF(modelo, cidade)
######################################################################################################################################
