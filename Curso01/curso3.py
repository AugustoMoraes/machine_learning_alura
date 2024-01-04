import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)

a_renomeaar = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizado'
}
dados = dados.rename(columns= a_renomeaar)
troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
print(dados.tail())

#sns.scatterplot(x='horas_esperadas', y='preco', data=dados)

#sns.scatterplot(x='horas_esperadas', y='preco', hue='finalizado', data=dados)
#sns.relplot(x='horas_esperadas', y='preco', hue='finalizado', col='finalizado', data=dados)
#plt.show()

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

SEED = 20 # determina a ordem dos números aleatórios, dessa forma mantém um padrão na divisão de treino e teste e a acurácio permanece a mesma
treino_x, teste_x, treino_y, teste_y = train_test_split(x,y,
                                                        random_state=SEED,test_size=0.25,
                                                        stratify= y) #utiliza 25% dos dados para treino e 75% pra teste
# stratify separa proporcionalmente os valores de treino, para que o modelo não aprenda só um resultado
modelo = LinearSVC()
treino = modelo.fit(treino_x, treino_y)
print(treino_y.value_counts())
print(teste_y.value_counts())

previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes)

print(f'A acurácioa foi {acuracia *100: .2f}%')

previsoes_de_base = np.ones(540)
acuracia_base = accuracy_score(teste_y, previsoes_de_base)
print(f'A acurária do algoritimo de baseline foi {acuracia_base * 100: .2f}%')

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()

y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixel = 100
eixo_x = np.arange( x_min, x_max, (x_max - x_min) / pixel )
eixo_y = np.arange( y_min, y_max, (y_max - y_min) / pixel )
# print(eixo_x)
# print(eixo_y)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

z = modelo.predict(pontos)
z2 = z.reshape(xx.shape)
print(z2)

plt.contour(xx, yy, z2, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)
plt.show()

