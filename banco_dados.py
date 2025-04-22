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
_LOCAL = "SIFAPSC"
### Encaminhamento aos Diretórios
if _LOCAL == "GH":
	caminho_dados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/dados/"
	caminho_resultados = "https://github.com/matheusf30/RS_saude_precisao/tree/main/resultados/porto_alegre/"
	caminho_modelos = "https://github.com/matheusf30/RS_saude_precisao/tree/main/modelos/"
elif _LOCAL == "SIFAPSC":
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
meteoro = "meteo_poa_h_96-22.csv"
bio = "obito_cardiovasc_total_poa_96-22.csv"
#oc = "PoA_oc_1996-01-01_2022-12-31.xlsx"
temps = "dados_83967_D_1996-01-01_2022-12-31.csv"
"""
prec = "prec_semana_ate_2023.csv"
tmin = "tmin_semana_ate_2023.csv"
tmed = "tmed_semana_ate_2023.csv"
tmax = "tmax_semana_ate_2023.csv"
"""
### Abrindo Arquivo
meteoro = pd.read_csv(f"{caminho_dados}{meteoro}", skiprows = 10, sep = ";", low_memory = False)
bio = pd.read_csv(f"{caminho_dados}{bio}", low_memory = False)
temps = pd.read_csv(f"{caminho_dados}{temps}", skiprows = 10, sep = ";", low_memory = False)
"""
prec = pd.read_csv(f"{caminho_dados}{prec}")
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
"""
print(f"\n{green}METEORO:\n{reset}{meteoro}\n")
print(f"\n{green}BIO:\n{reset}{bio}\n")
print(f"\n{green}TEMPERATURAS:\n{reset}{temps}\n")


#sys.exit()
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

#METEOROLOGIA NOVO
temps = temps.drop(columns = "Unnamed: 3")
temps = temps.rename(columns = {"Data Medicao" : "data",
							"TEMPERATURA MAXIMA, DIARIA(°C)" : "tmax",
							"TEMPERATURA MINIMA, DIARIA(°C)" : "tmin"})
temps["data"] = pd.to_datetime(temps["data"])
#temps[["tmax", "tmin"]] = temps[["tmax", "tmin"]].str.replace(',', '.').astype(float)
temps[["tmax", "tmin"]] = temps[["tmax", "tmin"]].applymap(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else float(x))
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
meteoro.dropna(axis = 0, inplace = True)
colunas_objt = meteoro.select_dtypes(include='object').columns
meteoro = meteoro.replace("," , ".")
meteoro[colunas_objt] = meteoro[colunas_objt].apply(lambda x: x.str.replace(",", "."))
meteoro["prec"] = pd.to_numeric(meteoro["prec"], errors = "coerce")
meteoro["pressao"] = pd.to_numeric(meteoro["pressao"], errors = "coerce")
meteoro["temp"] = pd.to_numeric(meteoro["temp"], errors = "coerce")
meteoro["ventovel"] = pd.to_numeric(meteoro["ventovel"], errors = "coerce")
meteoro = meteoro.groupby("data").filter(lambda x: len(x) == 3)
print(meteoro)
#sys.exit()
prec = meteoro.groupby(by = ["data"])["prec"].sum()
tmax = meteoro.groupby(by = ["data"])["temp"].max()
tmin = meteoro.groupby(by = ["data"])["temp"].min()
urmin = meteoro.groupby(by = ["data"])["umidade"].min()
urmax = meteoro.groupby(by = ["data"])["umidade"].max()
meteoro = meteoro.groupby(by = "data")[["pressao", "temp", "umidade", "ventodir", "ventovel"]].mean().round(2)
print(tmin)
meteoro["prec"] = prec
meteoro["tmin_3h"] = tmin
meteoro["tmax_3h"] = tmax
meteoro["urmin"] = urmin
meteoro["urmax"] = urmax
meteoro["amplitude_t_3h"] = meteoro["tmax_3h"] - meteoro["tmin_3h"]
print(f"\n{green}METEORO:\n{reset}{meteoro}\n")
print(f"\n{green}BIO:\n{reset}{bio}\n")
print(f"\n{green}TEMPERATURAS:\n{reset}{temps}\n")
#sys.exit()

# BIOMETEORO
meteoro.reset_index(inplace = True)
meteoro["data"] = pd.to_datetime(meteoro["data"])
total = total.to_frame(name = "obito")
total.reset_index(inplace = True)
biometeoro = meteoro.merge(total, on = "data", how = "inner")
biometeoro = biometeoro.merge(temps, on = "data", how = "inner")
#biometeoro[["tmax", "tmin"]] = biometeoro[["tmax", "tmin"]].astype(float)
biometeoro["amplitude_t"] = biometeoro["tmax"] - biometeoro["tmin"]
biometeoro = biometeoro[["data", "obito",
						"tmin_3h", "tmin", "temp", "tmax_3h", "tmax",
						"amplitude_t_3h", "amplitude_t","urmin", "umidade", "urmax",
						"prec", "pressao", "ventodir", "ventovel"]]
### ÍNDICES SIMPLIFICADOS DE SENSAÇÃO TÉRMICA
# Wind Chill #https://www.meteoswiss.admin.ch/weather/weather-and-climate-from-a-to-z/wind-chill.html
# Farenheit # miles per hour # https://www.weather.gov/media/epz/wxcalc/windChill.pdf
biometeoro["ventovel"] = biometeoro["ventovel"] * 3.6 # m/s >> km/h
Tc = biometeoro["temp"] # T (C) < 10
Tf = ((biometeoro["temp"] * 9) / 5) + 32  # C >> F
Tf_WC = ((biometeoro["tmin"] * 9) / 5) + 32  # C >> F 
Tf_HI = ((biometeoro["tmax"] * 9) / 5) + 32  # C >> F 
Vkmh = biometeoro["ventovel"] * 3.6 # ms >> km/h # V (km/h) > 4.82
Vmph = biometeoro["ventovel"] * 2.237 # ms >> mph 
RH = biometeoro["umidade"] # %
#meteoro["wind_chillC"] = 13.12 + 0.6215 * Tc - 11.37 * Vkmh**0.16 + 0.3965 * Tc * Vkmh**0.16 # C km/h # Fazer condicionantes
biometeoro["wind_chill"] = 35.74 + 0.6215 * Tf_WC - 35.75 * Vmph**0.16 + 0.4275 * Tf_WC * Vmph**0.16 # F mph # Fazer condicionantes
#meteoro["wind_chill"] = np.where((Tc < 10) & (Vkmh > 4.82),  meteoro["wind_chill"], None)
# Heat Index #
biometeoro["heat_index"] =  -42.379 + 2.04901523*Tf_HI + 10.14333127*RH - .22475541*Tf_HI*RH - .00683783*Tf_HI*Tf_HI - .05481717*RH*RH + .00122874*Tf_HI*Tf_HI*RH + .00085282*Tf_HI*RH*RH - .00000199*Tf_HI*Tf_HI*RH*RH
ajuste1 = ((13 - RH) / 4) * np.sqrt((17 - np.absolute(Tf_HI - 95.)) / 17)
ajuste2 =  ((RH - 85) / 10) * ((87 - Tf_HI) / 5)
#meteoro["heat_index"] = np.where((RH <= 13) & (Tf >= 80) & (Tf <= 112),  meteoro["heat_index"] - ajuste1, None)
#meteoro["heat_index"] = np.where((RH >= 85) & (Tf >= 80) & (Tf <= 87),  meteoro["heat_index"] - ajuste2, None)
biometeoro = biometeoro[["data", "obito", "heat_index", "wind_chill",
						"tmin_3h", "tmin", "temp", "tmax_3h", "tmax",
						"amplitude_t_3h", "amplitude_t","urmin", "umidade", "urmax",
						"prec", "pressao", "ventodir", "ventovel"]]
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
print(f"\n{green}BIOMETEORO:\n{reset}{biometeoro}\n")

biometeoro.to_csv(f"{caminho_dados}biometeoro_{_cidade}.csv", index = False)
print(f"\n{green}SALVO COM SUCESSO:\n{caminho_dados}biometeoro_{_cidade}.csv\n{reset}")

biometeoro_inverno = biometeoro.copy()
biometeoro_inverno.set_index("data", inplace = True)
print(f"\n{green}BIOMETEORO (INVERNO):\n{reset}{biometeoro_inverno}v\n")
biometeoro_inverno = biometeoro_inverno[(biometeoro_inverno.index.month >= 6) & (biometeoro_inverno.index.month <= 8)]
#biometeoro_inverno = biometeoro_inverno[biometeoro_inverno["data"].dt.month.isin(["06", "07", "08"])]
print(f"\n{green}BIOMETEORO (INVERNO):\n{reset}{biometeoro_inverno}v\n")
#sys.exit()
biometeoro_inverno.to_csv(f"{caminho_dados}biometeoro_inverno_{_cidade}.csv", index = False)
print(f"\n{green}SALVO COM SUCESSO:\n{caminho_dados}biometeoro_inverno_{_cidade}.csv\n{reset}")

biometeoro_verao = biometeoro.copy()
biometeoro_verao.set_index("data", inplace = True)
print(f"\n{green}BIOMETEORO (VERÃO):\n{reset}{biometeoro_verao}v\n")
biometeoro_verao1 = biometeoro_verao[(biometeoro_verao.index.month >= 12)]
biometeoro_verao2 = biometeoro_verao[(biometeoro_verao.index.month <= 2)]
biometeoro_verao = pd.concat([biometeoro_verao1, biometeoro_verao2])
biometeoro_verao.reset_index(inplace = True)
biometeoro_verao["data"] = pd.to_datetime(biometeoro_verao["data"])
biometeoro_verao.sort_values(by = "data", inplace = True)
print(f"\n{green}BIOMETEORO (VERÃO):\n{reset}{biometeoro_verao}v\n")
#sys.exit()
biometeoro_verao.to_csv(f"{caminho_dados}biometeoro_verao_{_cidade}.csv", index = False)
print(f"\n{green}SALVO COM SUCESSO:\n{caminho_dados}biometeoro_verao_{_cidade}.csv\n{reset}")


