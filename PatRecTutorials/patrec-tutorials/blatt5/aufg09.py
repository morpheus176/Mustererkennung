
import matplotlib.pyplot as plt
from common.classification import NeuralNetwork, ClassificationEvaluator
from common.data_provider import DataProvider
from common import visualization

def aufg09():
    #
    # In dieser Aufgabe geht es um die Klassifikation mit neuronalen Netzen.
    #
    # Dazu sollen Sie ein einschichtiges (kein Hidden Layer) neuronales Netz
    # zur Klassifikation auf dem data2d Datensatz verwenden.
    # Ueberlegen Sie zunaechst welche Klassifikationsprobleme sich mit dieser
    # Netzwerk-Topologie loesen lassen und waehlen Sie den Trainings- und
    # Testdatensatz entsprechend.
    #
    # Implementieren Sie nun Training und Test fuer ein neuronales Netz, welches
    # zwei Eingabe und ein Ausgabeneuron enthaehlt. Beruecksichtigen Sie auch das bias-Gewicht.
    # Verwenden Sie sowohl eine lineare als auch eine nicht-lineare Aktivierungsfunktion
    # im Ausgabeneuron.
    #
    # Visualisieren Sie den mittleren quadratischen Fehler ueber alle Iterationen.
    # Visualisieren Sie das neuronale Netz ueber einige (optional alle) Iterationen durch
    # Methode die plot_hyperplane im Modul common.visualization. Orientieren Sie sich an der
    # Implementierung von plot_svm. Fuer die Erzeugung passender Funktionsobjekte koennen
    # anonyme (lambda) Funktionen hilfreich sein.
    #
    # Hinweise:
    # - Das oben beschriebene neuronale Netz laesst sich durch eine Matrixmultiplikation
    #   auswerten.
    # - Beim batch Training koennen Sie pro Iterationsschritt die Aenderung der Gewichte
    #   ueber alle Samples mitteln.
    # - Waehlen Sie die Lernrate nicht zu gross.
    # - Den Klassenrumpf finden Sie im Modul common.classification.
    #
    #
    # Was ist bei der Initialisierung des Trainings zu beachten?
    # Was ist bei der Terminierung des Trainings zu beachten?
    #    Optional: Verwenden Sie eine Validierungsstichprobe.
    # Welchen Zweck erfuellt das bias-Gewicht?
    # Wie wirken sich unterschiedliche Lernraten aus?
    # Wie wirken sich unterschiedliche Aktivierungsfunktionen aus?
    # Wie verhalten sich die Ergebnisse im Vergleich zu einer linearen 2-Klassen SVM?

    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)

    VISUALIZE = True
    if VISUALIZE:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = train_data_provider.get_class_arr(0)
        ax.scatter(data[:,0], data[:,1], c='#FF0000', edgecolor=(0, 0, 0))
        data = train_data_provider.get_class_arr(1)
        ax.scatter(data[:,0], data[:,1], c='#00FF00', edgecolor=(0, 0, 0))
        data = train_data_provider.get_class_arr(2)
        ax.scatter(data[:,0], data[:,1], c='#0000FF', edgecolor=(0, 0, 0))
    plt.show()

    # Zwei Klassen, welche mit einem einschichtigen neuronalen Netz separierbar sind, auswaehlen
    raise NotImplementedError('Implement me')

    # Waehlen Sie eine geeignete Lernrate und die Anzahl von Trainingsiterationen
    lr = 0
    max_iterations = 0

    nn = NeuralNetwork(learning_rate=lr, iterations=max_iterations, activation='Linear')
    nn.estimate(train_data, train_labels)
    nn_result_labels = nn.classify(test_data, iteration=100)

    #
    # Nutzen Sie zur Evaluation der Ergebnisse die Klasse ClassificationEvaluator
    # im Modul common.classifiers und visualisieren Sie die Ergebnisse.

    raise NotImplementedError('Implement me')

    #
    # Welche Eigenschaften hat ein neuronales Netz mit dem sich beliebige 2-Klassenprobleme
    # auf dem data2d Datensatz loesen lassen?
    # Wie wuerde man ein solches Netz fuer die Loesung des 3-Klassenproblems erweitern?
    #
    # Optional:
    #
    # Implementieren Sie ein neuronales Netz, mit dem sich beliebige 2-Klassenprobleme auf
    # dem data2d Datensatz loesen lassen. Visualisieren Sie den Trainingprozess.
    #
    # Erweitern Sie Ihr Netz zur Loesung des 3-Klassenproblems.
    #

if __name__ == '__main__':
    aufg09()
