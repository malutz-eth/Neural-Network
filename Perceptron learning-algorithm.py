import matplotlib.pyplot as plt


from random import choice
from numpy import array,dot,zeros,random

#als treshold funktion wurde hier die Lambda funktion ausgewählt, auch rectangle function genannt. 
heaviside = lambda x: 0 if x < 0 else 1

#training function, which is responsible for the change of w
def fit(iterations, training_data_set,w):
    errors = []
    weights= []
    for i in range(iterations):
    #zufällige Auswahl eines Trainingsbeispiel
        training_data= choice(training_data_set) #choice returns a choosen element of the training_data_set
        x=training_data[0]
        y=training_data[1]
#output ermitteln
        y_hat = heaviside(dot(w,x))
        error = y - y_hat
        #Hier werden die Fehler gesammelt
        errors.append(error) #append function adds elements to a list
        weights.append(w)
        #Anpassung der Gewichtung
        w +=error*x
    return errors, weights

def main(): #Hauptprogram
    #the main function put all steps and fucntions needed together

    training_data_set=[(array([1,0,0]),0),
                        (array([1,0,1]),1),
                       (array([1,1,0]),1),
                       (array([1,1,1]),1)]
    #Irgend ein Wert für die Reproduzierbarkeit der Ergebnisse
    random.seed(1100000)
    w=zeros(3) #Array of the lenght 2 with zeros
    iterations = 10000

    errors,weights = fit(iterations,training_data_set,w)
    w = weights[iterations-1]
    print("Weights vector at the end of training: ")
    print(w)

    #graphic of expamples
    print("Auswertung am Ende des Trainings: ")
    for x,y in training_data_set:
        y_hat = heaviside(dot(x,w))
        print("{}: {} -> {}".format(x,y,y_hat))
    fignr = 1

    plt.figure(fignr,figsize=(10,10))
    plt.plot(errors)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Iterations')
    plt.ylabel(r"$(y-\hat y)$")
    plt.show()

main()
