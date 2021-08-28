import numpy as np

# On met nos valeurs d'entrées dans un tableau x_entrer
# i= longueur, j= largeur
x_entrer = np.array(([1, 1.5], [2, 1], [4, 1.5], [1, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [4.5, 1],),
                    dtype=float)

# On met nos valeur de déduction dans un tableau y_sortie
y_sortie = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), dtype=float)
# donnée de sortie 1 = rouge et donnée de sortie 0 = bleu

# On met toutes nos valeurs d'entrées sur la meme echelle c'est a dire entre 0 et 1 en appliquant la fonction sigmoid
x_entrer = x_entrer/np.max(x_entrer, axis=0)

x = np.split(x_entrer, [8])[0] # On recupére seulement les valeurs dont on connait la couleur
xPrediction = np.split(x_entrer, [8])[1] # On recupére la valeur dont on ne connait la couleur

class Neural_Network(object):

    def __init__(self):
        self.inputSize = 2 # 2 represente le nombre de valeur en entré (L, l)
        self.outputSize = 1 # 1 represente le nombre de valeur en sortie (la couleur predite)
        self.hiddenSize = 3 # 3 represente le nombre de neurone cachés
        # Generation aléatoir des poids
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)  # On aura une matrice 2x3
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) # On aura une matrice 3x1

    # la fonction nous permet de multiplier nos valeurs d'entrés et leur poids
    def forward(self, x):
        self.z = np.dot(x, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        o = self.sigmoid(self.z3) # on determine la veuleur de sortie en appliquant la fonction sigmoid a z3
        return o

    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def sigmoidPrime(self, s): # on calcul la derivé de la fonction sigmoid
        return s * (1 - s)

    def backword(self, x, y, o):
        self.error = y - o # on calcul l'erreur du neurone de sortie
        self.o_delta = self.error * self.sigmoidPrime(o) # On calcul l'erreur delta du neurone de sortie
        self.z2_error = self.o_delta.dot(self.w2.T) # on calcul de nos 3 neurones cachés
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2) # on calcul l'erreur delta des 3 neurones cachés
        self.w1 += x.T.dot(self.z2_delta)  # on calcul l'erreur des 2 neurones d'entrés
        self.w2 += self.z2.T.dot(self.o_delta) # On calcul l'erreur delta des 2 neurones d'entrés

        # fonction d'entainement
    def train(self, x, y):
        o = self.forward(x)
        self.backword(x, y, o)

    def prediction(self):
        print("Données prédite aprés entrainnement: ")
        print("Entrées : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) > 0.5):
            print("La fleur est rouge \n")
        else:
            print("La fleur est bleue \n")

MN = Neural_Network()
for i in range(30000):
    print("#" + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(x))
    print("Sortie actuelle: \n" + str(y_sortie))
    print("Sortie predite: \n" + str(np.matrix.round(MN.forward(x), 2)))
    MN.train(x, y_sortie)

MN.prediction()



