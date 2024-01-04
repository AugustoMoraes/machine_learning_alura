import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)

mapa = {
    'home': 'principal',
    'how_it_works': 'como_funciona',
    'contact': 'contato',
    'bought' : 'comprou'
}
dados = dados.rename(columns= mapa)

x = dados[['principal','como_funciona', 'contato']]
y = dados['comprou']

# treino_x = x[:75] #pega os elementos de 0 a 74 / 75 elementos
# treino_y = y[:75]
# teste_x = x[75:] # pega os elementos 75 até o ultimo / 24 elementos
# teste_y = y[75:]
# print(x.head())
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

