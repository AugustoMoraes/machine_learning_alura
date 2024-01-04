from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# features (1 sim, 0 não)
# pelo longo?
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, dog1, dog2, dog3]
treino_y = [1,1,1,0,0,0]



model = LinearSVC()
model.fit(treino_x, treino_y) # aprender com os dados e em qual classe ele pertence

animal_misterioso = [1,1,1]

resullt = model.predict([animal_misterioso])

misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]

testes_x = [misterio1,misterio2,misterio3]
testes_y = [0,1,1]

previsoes = model.predict(testes_x)

corretos = (previsoes == testes_y).sum()
total = len(testes_x)
taxa_acertos = corretos / total

acuracia = accuracy_score(testes_y, previsoes)
print(f'A taxa de acerto é:  ${taxa_acertos*100: .2f}')
print(f'Accuracy: ${acuracia * 100: .2f}%')
