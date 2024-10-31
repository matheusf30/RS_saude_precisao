# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

#########################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/dados/"
caminho_indices = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/indices/"
caminho_resultados = "/home/sifapsc/scripts/matheus/RS_saude_precisao/resultados/porto_alegre/"
caminho_modelos = "/home/sifapsc/scripts/matheus/RS_saude_precisao/modelos/"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos
dataset1 = "PORTO_ALEGRE_dataset1.csv"
dataset2 = "PORTO_ALEGRE_dataset2.csv"
dataset3 = "PORTO_ALEGRE_dataset3.csv"
dataset1 = pd.read_csv(f"{caminho_dados}{dataset1}", low_memory = False)
dataset2 = pd.read_csv(f"{caminho_dados}{dataset2}", low_memory = False)
dataset3 = pd.read_csv(f"{caminho_dados}{dataset3}", low_memory = False)
print(f"\n{green}dataset1:\n{reset}{dataset1}\n")
print(f"\n{green}dataset2:\n{reset}{dataset2}\n")
print(f"\n{green}dataset3:\n{reset}{dataset3}\n")
### Seleção de Períodos
dataset_periodo1 = dataset1.copy()
dataset_periodo1["data"] = pd.to_datetime(dataset_periodo1["data"], errors = "coerce")
dataset_periodo1.set_index("data", inplace = True)
print(f"\n{red}PERÍODO {green}NÃO {red}SELECIONADO:\n{reset}{dataset_periodo1}\n{dataset_periodo1.info()}\n")
dataset_quente1 = dataset_periodo1[(dataset_periodo1.index.month >= 10) & (dataset_periodo1.index.month <= 3)]
dataset_frio1 = dataset_periodo1[(dataset_periodo1.index.month >= 4) & (dataset_periodo1.index.month <= 9)]
#dataset_periodo1.reset_index(inplace = True)
#dataset_periodo1.drop(columns = "dia", inplace = True)
#dataset_periodo1.set_index("data", inplace = True)
print(f"\n{green}PERÍODO {red}QUENTE {green}SELECIONADO:\n{reset}{dataset_quente1}\n{dataset_quente1.info()}\n")
print(f"\n{green}PERÍODO {red}FRIO {green}SELECIONADO:\n{reset}{dataset_frio1}\n{dataset_frio1.info()}\n")

