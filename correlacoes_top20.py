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

### Concatenando CIDs e Meteorologia
meteoro1_in = meteoro.merge(iam1, on = "data", how = "inner")
meteoro2_in = meteoro.merge(iam2, on = "data", how = "inner")
meteoro3_in = meteoro.merge(iam3, on = "data", how = "inner")
meteoro4_in = meteoro.merge(iam4, on = "data", how = "inner")
print(f"\n{green}IAM1 INNER\n{reset}{meteoro1_in}\n")
print(f"\n{green}IAM2 INNER\n{reset}{meteoro2_in}\n")
print(f"\n{green}IAM3 INNER\n{reset}{meteoro3_in}\n")
print(f"\n{green}IAM4 INNER\n{reset}{meteoro4_in}\n")
meteoro1_out = meteoro.merge(iam1, on = "data", how = "outer").fillna(0)
meteoro2_out = meteoro.merge(iam2, on = "data", how = "outer").fillna(0)
meteoro3_out = meteoro.merge(iam3, on = "data", how = "outer").fillna(0)
meteoro4_out = meteoro.merge(iam4, on = "data", how = "outer").fillna(0)
print(f"\n{green}IAM1 OUTER\n{reset}{meteoro1_out}\n")
print(f"\n{green}IAM2 OUTER\n{reset}{meteoro2_out}\n")
print(f"\n{green}IAM3 OUTER\n{reset}{meteoro3_out}\n")
print(f"\n{green}IAM4 OUTER\n{reset}{meteoro4_out}\n")
print(f"\n{green}IAM4 OUTER (COLUMNS)\n{reset}{meteoro4_out.columns}\n")







