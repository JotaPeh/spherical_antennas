I - Autoria:
    Os códigos foram desenvolvidos por João Pedro Falcão da Silva como parte do projeto de Análise e Projeto de Antenas
de Microfita Esféricas com Polarização Circular, orientado pelo Prof. Dr. Daniel Basso Ferreira (ITA) para o trabalho de
conclusão do curso de Engenharia Eletrônica no Instituto Tecnológico de Aeronáutica. Esses scripts servem para a análise
e síntese de antenas de microfita esféricas com diferentes configurações de polarização (circular e linear) e outras
variações, explorando métodos como o da cavidade ressonante e o método da corrente magnética.


II - Dependências:
Versão utilizada: Python 3.11.6
Instalação dos requisitos: pip install -r requirements.txt


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
    Para as sínteses, um .txt será gerado ao final com os parâmetros projetados em Resultados->Param, além de algumas
figuras com o patch projetado. Para as análises, uma pasta com o nome da análise realizada será gerada dentro de
Resultados, contendo as figuras de saída. Ademais, um .txt com os valores de ganho também pode ser gerado ao final.


V - Ajustes:
    O usuário pode ajustar as variáveis no próprio código antes da execução. Algumas variáveis importantes são:
show = 1: Mostra resultados durante execução do código
test = 1: Útil para debugar as funções no código Síntese_CP.py
project = 1: Liga o modo de projeto completo em Síntese_CP.py, do qual se descobre o valor de 'p', quando desligado,
             pode-se escolher um valor de 'p' para ser projetado mais rapidamente
flm_des: nas sínteses, frequência central desejada de projeto (ex.: 1575.42e6  # Hz).
modo: modo da antena linearmente polarizada a ser projetada na Síntese_LP.py (ex.: '10' ou '01').
dtc: nas análises, deslocamento do centro do patch em theta (em radianos), como dtc = 1 * dtr.
dpc: nas análises, deslocamento do centro do patch em phi (em radianos), como dpc = 1 * dtr.