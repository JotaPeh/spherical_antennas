## Instructions in English

- Authorship: The codes were developed by João Pedro Falcão da Silva as part of the project Analysis and Design of Spherical
Microstrip Antennas with Circular Polarization, supervised by Prof. Dr. Daniel Basso Ferreira (ITA) as the final project for
the Electronic Engineering course at the Aeronautics Institute of Technology (ITA). These scripts are designed for the analysis
and synthesis of spherical microstrip antennas with various polarization configurations (circular and linear) and other
variations, exploring methods such as the resonant cavity method and the magnetic current method. The full version of the work
can be accessed for more details at: https://1drv.ms/b/s!AiJsFG-QQveHh8Zi64OE_2u-dKuhMw?e=J3NySy

- Dependencies:
Version used: Python 3.11.6
Installation of requirements: pip install -r requirements.txt
If there is an issue saving images: pip install -U kaleido

- Files:
Análise_Cavidade.py: Simulation based on the resonant cavity model.
Análise_LP.txt: Analysis results for linear polarization.
Análise_CP.py: Analysis of circularly polarized antennas.
Análise_CP_2P.py: Analysis of circularly polarized antennas fed by two probes.
Análise_CP_Trunc.py: Analysis of circularly polarized antennas with truncated corners.
Auxiliar_CP_2P.py: Analysis helper for the two-probe antenna fed via a 90° hybrid.
Síntese_LP.py: Synthesis of linearly polarized antennas.
Síntese_CP.py: Synthesis of circularly polarized antennas.
Síntese_CP_2P.py: Synthesis of circularly polarized antennas with two probes.
Síntese_CP_Trunc.py: Synthesis of circularly polarized antennas with truncated corners.

- Instructions:
  To run any script, use Python with the appropriate libraries installed (NumPy, SciPy, Matplotlib). Adjust the variables as
needed, as shown in the following options, run the desired script in the terminal, e.g., python Análise_CP.py, and, for
analysis scripts, input the file with the parameters to be analyzed. Ensure the HFSS folder contains the necessary results
from antenna simulations in the HFSS software, typically in .csv format, for the proper generation of comparative images in
the analyses.
  For the synthesis scripts, a file *_Param.txt will be generated in the Resultados->Param folder with the designed parameters,
along with some figures showing the designed patch. For the analysis scripts, a folder named after the performed analysis will
be created inside Resultados, containing the output figures. Additionally, a file *_Ganho.txt with gain values may be
generated at the end.

- Adjustments:
  Users can adjust variables in the code before execution. Some important variables include:
show = 1: Displays results during code execution; Smith chart figures open in the browser.
test = 1: Useful for debugging root-finding functions in Síntese_CP.py.
project = 1: Enables full project mode in Síntese_CP.py, discovering the value of 'p'; when disabled, allows the selection
             of a 'p' value for faster design.
normalizado = 0: In Análise_CP_Trunc.py, toggles far-field plots to the unnormalized version.
path = HFSS/CP_pre/: Displays results prior to fringe field correction in Análise_CP.py and Análise_LP.py.
flm_des: In synthesis scripts, desired central design frequency (e.g., 1575.42e6  # Hz).
modo: Mode of the linearly polarized antenna to be designed in Síntese_LP.py (e.g., '10' or '01').
dtc: In analysis scripts, displacement of the patch center in theta (in radians), e.g., dtc = 1 * dtr.
dpc: In analysis scripts, displacement of the patch center in phi (in radians), e.g., dpc = 1 * dtr.
l = m = 35: Number of modes considered in far-field summations, with the constraint m <= l.

First Iteration in Síntese_CP_Trunc.py: Import results from the HFSS linear antenna for scaling Dphi, Dthetaa, the Band, and the angular position of the probe thetap.

- Simulations:
  Link to the .aedt and .aedb files used to compare the code results with HFSS simulations: https://1drv.ms/f/s!AiJsFG-QQveHh7944sCgQLuEOozPFA?e=muwdAs

#############################################################################################################################################
## Instruções em português

I - Autoria:
    Os códigos foram desenvolvidos por João Pedro Falcão da Silva como parte do projeto de Análise e Projeto de Antenas
de Microfita Esféricas com Polarização Circular, orientado pelo Prof. Dr. Daniel Basso Ferreira (ITA) para o trabalho de
conclusão do curso de Engenharia Eletrônica no Instituto Tecnológico de Aeronáutica. Esses scripts servem para a análise
e síntese de antenas de microfita esféricas com diferentes configurações de polarização (circular e linear) e outras
variações, explorando métodos como o da cavidade ressonante e o método da corrente magnética. A versão completa do trabalho
pode ser acessada para mais detalhes em: https://1drv.ms/b/s!AiJsFG-QQveHh8Zi64OE_2u-dKuhMw?e=J3NySy


II - Dependências:
Versão utilizada: Python 3.11.6
Instalação dos requisitos: pip install -r requirements.txt
Caso haja problema no salvamento de imagens: pip install -U kaleido


III - Arquivos:
Análise_Cavidade.py: Simulação baseada no modelo da cavidade ressonante.
Análise_LP.txt: Resultados de análise para polarização linear.
Análise_CP.py: Análise de antenas de polarização circular.
Análise_CP_2P.py: Análise de antenas com polarização circular alimentadas por duas pontas de prova.
Análise_CP_Trunc.py: Análise de antenas com polarização circular e cantos truncados.
Auxiliar_CP_2P.py: Auxiliar de análise para a antena com duas pontas de prova alimentada via híbrida de 90°.
Síntese_LP.py: Síntese de antenas de polarização linear.
Síntese_CP.py: Síntese de antenas de polarização circular.
Síntese_CP_2P.py: Síntese de antenas com duas pontas de prova e polarização circular.
Síntese_CP_Trunc.py: Síntese de antenas com polarização circular e cantos truncados.


IV - Instruções:
    Para executar qualquer código, use o Python com as bibliotecas apropriadas instaladas (NumPy, SciPy, Matplotlib).
Basta ajustar as variáveis da maneira desejada, como mostrado nas opções seguintes, rodar o script desejado no terminal,
por exemplo "python Análise_CP.py", e, em seguida, no caso das análises, digitar o arquivo com os parâmetros a serem
analisados. Certifique-se de que a pasta HFSS contenha os resultados necessários advindos de simulações das antenas no
software HFSS, geralmente em .csv, para a apropriada geração de imagens comparativas nas análises.
    Para as sínteses, um arquivo *_Param.txt será gerado ao final com os parâmetros projetados em Resultados->Param, além
de algumas figuras com o patch projetado. Para as análises, uma pasta com o nome da análise realizada será gerada dentro
de Resultados, contendo as figuras de saída. Ademais, um arquivo *_Ganho.txt com os valores de ganho também pode ser
gerado ao final.


V - Ajustes:
    O usuário pode ajustar as variáveis no próprio código antes da execução. Algumas variáveis importantes são:
show = 1: Mostra resultados durante execução do código, as figuras da carta de Smith abrem no navegador.
test = 1: Útil para debugar as funções de busca de raízes no código Síntese_CP.py.
project = 1: Liga o modo de projeto completo em Síntese_CP.py, do qual se descobre o valor de 'p', quando desligado,
             pode-se escolher um valor de 'p' para ser projetado mais rapidamente.
normalizado = 0: Em Análise_CP_Trunc.py, alterna gráficos de campo distante para versão desnormalizada.
path = 'HFSS/CP_pre/': Mostra resultados anteriores à correção dos campos de franja em Análise_CP.py e Análise_LP.py.
flm_des: Nas sínteses, frequência central desejada de projeto (ex.: 1575.42e6  # Hz).
modo: Modo da antena linearmente polarizada a ser projetada na Síntese_LP.py (ex.: '10' ou '01').
dtc: Nas análises, deslocamento do centro do patch em theta (em radianos), como dtc = 1 * dtr.
dpc: Nas análises, deslocamento do centro do patch em phi (em radianos), como dpc = 1 * dtr.
l = m = 35: Número de modos considerados nos somatórios de campo distante, com a restrição m <= l

Primeira Iteração em Síntese_CP_Trunc.py: Trazer resultados da antena linear do HFSS para o escalonamento de Dphia
Dthetaa, além da banda Band e da posição agular da ponta de prova thetap.

VI - Simulações:
Link para os arquivos .aedt e .aedb utilizados para comparar os resultados do código com as simulações no HFSS:
https://1drv.ms/f/s!AiJsFG-QQveHh7944sCgQLuEOozPFA?e=muwdAs
