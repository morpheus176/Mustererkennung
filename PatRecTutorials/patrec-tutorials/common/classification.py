from collections import defaultdict
import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import cdist
from common import log_math
import operator
from scipy.stats import multivariate_normal


class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern

        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string),
                siehe auch scipy Funktion cdist
        """
        self.k = k_neighbors
        self.metric = metric
        #raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.

        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden.
        Diskutieren Sie, was den K-NN stattdessen definiert.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        self.training_data = train_samples
        self.training_labels = train_labels
        #raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Implementieren Sie die Klassifikation der Daten durch den KNN.
        #
        # Nuetzliche Funktionen: scipy.spatial.distance.cdist, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        
        prediction = []
        
        for i in range(test_samples.shape[0]): # für jeden zu klassifizierenden Datenpunkt
            stats = {'a': 0, 'b': 0, 'c': 0}
            a = test_samples[i] # keine Veränderung von Arrays in For Loops in Python, deswegen ein temporäres Objekt
            if len(a.shape) != 2: # mache eben aus a ein 2d-Array mit nur einer Zeile, falls es noch keins ist.
                a = a.reshape(1, a.shape[0])
            distanceTest = scipy.spatial.distance.cdist(a, self.training_data, self.metric)[0] # Hierfür MUESSEN beide Arrays zweidimensional sein
            distanceList = np.stack((distanceTest, self.training_labels), axis=0).astype('float64') # Klebe die beiden Arrays als Spalten zusammen!
            #print(distanceList)
            sortedDistanceList = distanceList[:,distanceList[0,:].argsort()] # Sortiere nach der ersten Spalte!
            kNearestNeighbors = sortedDistanceList[:,:self.k] # Nimm nur die ersten k Werte -> k nächste Nachbarn
            for l in range(0, self.k):
                if kNearestNeighbors[1,l] == 0:
                    stats['a'] += 1
                elif kNearestNeighbors[1,l] == 1:
                    stats['b'] += 1
                else:
                    stats['c'] += 1
            if max(stats.items(), key=operator.itemgetter(1))[0] == 'a':
                prediction.append(0)
            elif max(stats.items(), key=operator.itemgetter(1))[0] == 'b':
                prediction.append(1)
            else:
                prediction.append(2) # Falls gleich viele Signale und Background in der Nachbarschaft liegen
                # können wir keine Entscheidung treffen...
        return np.asarray(prediction) # Gib ein numpy Array zurück, weil man damit besser arbeiten kann als mit einer Liste
        #raise NotImplementedError('Implement me')


class GaussianClassifier(object):

    def __init__(self):
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.
        """

        self.mean_list = []
        self.cov_list = []
        self.p_k = []
        self.prob_k = []

        # raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        """Erstellt den Normalverteilungsklassikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """

        self.train_data = train_samples
        self.train_labels = train_labels

        self.labels = np.unique(self.train_labels)

        for label in self.labels:
            class_data = self.train_data[self.train_labels==label]

            mean = sum(class_data)/len(class_data)
            meanx = sum(class_data[:, 0])/len(class_data[:, 0])
            meany = sum(class_data[:, 1])/len(class_data[:, 1])

            sx = 1/(len(class_data[:, 0]) - 1) * sum((class_data[:, 0] - meanx)**2)
            sy = 1/(len(class_data[:, 1]) - 1) * sum((class_data[:, 1] - meany)**2)
            sxy = 1/(len(class_data[:, 0]) - 1) * sum((class_data[:, 0] - meanx) * (class_data[:, 1] - meany))

            cov = np.matrix([[sx, sxy], [sxy, sy]])

            self.mean_list.append(mean)
            self.cov_list.append(cov)

            p = len(class_data)/len(self.train_data)
            self.p_k.append(p)
        

        # raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Werten Sie die Dichten aus und klassifizieren Sie die
        # Testdaten.
        #
        # Hinweise:
        #
        # Durch welche geeignete monotone Transformation lassen sich numerische
        # Probleme bei der Auswertung von extrem kleinen Dichtewerten vermeiden?
        # Beruecksichtigen Sie das in Ihrer Implementierung.
        #
        # Erstellen Sie fuer die Auswertung der transformierten Normalverteilung
        # eine eigene Funktion. Diese wird in den folgenden Aufgaben noch von
        # Nutzen sein.
        self.prediction = []
        self.test_data = test_samples
        self.prob_list = np.zeros((len(self.test_data), 3))
        #self.total_list = []

        self.total = 0

        for i in range(0,len(self.labels)):

            # Berechnung der klassenbedingten Dichten
            prob = multivariate_normal.pdf(self.test_data, mean=self.mean_list[i], cov=self.cov_list[i])
            self.prob_list[:, i] = prob

            # Berechnung der totalen Wahrscheinlichkeiten
            self.total += prob * self.p_k[i]

        self.post = np.zeros((len(self.test_data), 3))
        
        for i in range(0, len(self.test_data)):

            for j in range(0, len(self.labels)):
                self.post[i, j] = self.p_k[j] * self.prob_list[i, j] / self.total[i]

            if self.post[i, :].argmax() == 0:
                self.prediction.append(self.labels[0])
            elif self.post[i, :].argmax() == 1:
                self.prediction.append(self.labels[1])
            else:
                self.prediction.append(self.labels[2])

        return np.asarray(self.prediction)


        # raise NotImplementedError('Implement me')

        

class MDClassifierClassIndep(object):

    def __init__(self, quantizer, num_densities):
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die verwendet
                werden sollen.
        """
        raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        """Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Diese Methode soll das Training eines Mischverteilungsklassifikators
        # mit klassenunabhaengigen Komponentendichten implementieren (siehe Skript S. 67 f.).
        #
        # Die folgenden Funtionen koennen bei der Implementierung von Nutzen
        # sein:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.slogdet.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html

        #
        # Schaetzen Sie das GMM der Trainingsdaten.
        #
        # Wieviele Datenpunkte werden zur Schaetzung jeder Normalverteilung mindestens
        # benoetigt und welche Eigenschaften muessen diese haben?
        # Beruecksichtigen Sie das in Ihrer Implementierung.

        raise NotImplementedError('Implement me')

        #
        # Bestimmen Sie fuer jede Klasse die spezifischen Mischungsgewichte.
        # Beachten Sie, dass die Dichteauswertung wieder ueber eine geeignete
        # monotome Transformationsfunktion geschehen soll. Verwenden Sie hierfuer
        # die Funktion, die Sie bereits fuer den GaussianClassifier implementiert
        #
        # Achten Sie darauf, dass sich die Mischungsgewichte zu 1 addieren.

        raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')

class MDClassifierClassDep(object):

    def __init__(self, quantizer, num_densities):
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die je Klasse
                verwendet werden sollen.
        """
        raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        """Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Schaetzen Sie die jeweils ein GMM fuer jede Klasse.
        #
        # Tipp: Verwenden Sie die bereits implementierte Klasse MDClassifierClassIndep

        raise NotImplementedError('Implement me')

    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError('Implement me')

class NeuralNetwork(object):

    def __init__(self, learning_rate, iterations, activation='Linear'):
        """Initialisiert das Perzeptron und die Klassen Variablen als Modellparameter.

        Params:
            learning_rate: Faktor zur Gewichtung der Gewichtsupdates.
            iterations: Anzahl der Trainingsiterationen.
            activation: String, definiert die Aktivierungsfunktion des Ausgabeneurons
                'Linear', 'TanH', 'Sigmoid', ...
        """
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        # Speichern Sie den Root-Mean-Squared Error fuer jede Trainingsiteration
        self.__nn_rms_list = []

        # Initialisieren Sie die Aktivierungsfunktionen und deren Ableitungen mithilfe des lambda Operators
        # https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions
        # if activation == 'Linear':
        #    self.__activation = lambda x: ...
        #    self.__activation_deriv = lambda x: ...
        raise NotImplementedError('Implement me')

    def estimate(self, train_samples, train_labels):
        """Trainiert das neuronale Netzwerk mit den gegebenen Trainingsdaten.

        Optimiert die Modellparameter basierend auf den Trainingsdaten
        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.

        Params:
            train_samples: ndarray, welches die Merkmalsvektorenthat zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-dimensional), Klassenlabels
                (d components, train_labels.shape=(d,) ).
        """
        raise NotImplementedError('Implement me')

    def classify(self, test_samples, iteration=None):
        """Klassifiziert die Testdaten
        Params:
            test_samples: ndarray mit Testdaten (Nxdim) mit N der Anzahl Daten und
               dim der Anzahl Dimensionen pro Sample
            iteration: Fuehrt die Klassifikation mit Gewichten einer bestimmten Iteration
               durch. (Nuetzlich fuer die Visualisierung). Default: None, nehme die Gewichte
               der letzten Iteration.
        """
        raise NotImplementedError('Implement me')

    @property
    def nn_rms_list(self):
        raise NotImplementedError('Implement me')

    def activation(self, x_input):
        """Berechnet Aktivierungsfunktion."""
        # Wird fuer unittest der Aktivierungsfunktion benoetigt
        return self.__activation(x_input)

    def activation_deriv(self, x_input):
        """Berechnet die Ableitung der Aktivierungsfunktion"""
        # Wird fuer unittest der Aktivierungsfunktion benoetigt
        return self.__activation_deriv(x_input)

class CrossValidation(object):

    def __init__(self, samples, labels, n_folds):
        """Initialisiert die Kreuzvalidierung

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
            labels: ndarray mit Labels fuer Beispieldaten, shape=(d,)
            n_folds: Anzahl Folds ueber die die Kreuzvalidierung durchgefuehrt
                werden soll.

        mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
        """
        self.__samples = samples
        self.__labels = labels
        self.__n_folds = n_folds

    def validate(self, classifier):
        """Fuert die Kreuzvalidierung mit dem Klassifikator 'classifier' durch.

        Params:
            classifier: Objekt, das folgende Methoden implementiert (siehe oben)
                estimate(train_samples, train_labels)
                classify(test_samples) --> test_labels

        Returns:
            crossval_overall_result: Erkennungsergebnis der gesamten Kreuzvalidierung
                (ueber alle Folds)
            crossval_class_results: Liste von Tuple (category, result) die klassenweise
                Erkennungsergebnisse der Kreuzvalidierung enthaelt.
        """
        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.__n_folds):
            train_samples, train_labels, test_samples, test_labels = self.samples_fold(fold_index)
            classifier.estimate(train_samples, train_labels)
            estimated_test_labels = classifier.classify(test_samples)
            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results

    @staticmethod
    def __crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result

    def samples_fold(self, fold_index):
        """Berechnet eine Aufteilung der Daten in Training und Test

        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.

        Returns:
            train_samples: ndarray mit Trainingsdaten, shape=(d_train,t)
            train_label: ndarray mit Trainingslabels, shape=(d_train,t)
            test_samples: ndarray mit Testdaten, shape=(d_test,t)
            test_label: ndarray mit Testlabels, shape=(d_test,t)

        mit d_{train,test} Beispieldaten und t dimensionalen Merkmalsvektoren.
        """
        n_samples = self.__samples.shape[0]
        test_indices = range(fold_index, n_samples, self.__n_folds)
        train_indices = [train_index for train_index in range(n_samples)
                             if train_index not in test_indices]

        test_samples = self.__samples[test_indices, :]
        test_labels = self.__labels[test_indices]
        train_samples = self.__samples[train_indices, :]
        train_labels = self.__labels[train_indices]

        return train_samples, train_labels, test_samples, test_labels

class ClassificationEvaluator(object):

    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis
        auf Testdaten.

        Params:
            estimated_labels: ndarray (1-Dimensional) mit durch den Klassifikator
                bestimmten Labels (N Komponenten).
            groundtruth_labels: ndarray (1-Dimensional) mit den tatsaechlichen
                Labels (N Komponenten).
        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        self.__binary_result_mat = groundtruth_labels == estimated_labels

    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.

        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        if mask is None:
            mask = np.ones_like(self.__binary_result_mat, dtype=bool)
        masked_binary_result_mat = self.__binary_result_mat[mask]
        n_samples = len(masked_binary_result_mat)
        n_correct = masked_binary_result_mat.sum()
        n_wrong = n_samples - n_correct
        error_rate = n_wrong / float(n_samples)
        error_rate *= 100
        return error_rate, n_wrong, n_samples

    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten

        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        category_list = sorted(set(self.__groundtruth_labels.ravel()))
        cat_n_err_list = []
        for category in category_list:
            category_mask = self.__groundtruth_labels == category
            err, n_wrong, n_samples = self.error_rate(category_mask)
            cat_n_err_list.append((category, err, n_wrong, n_samples))

        return cat_n_err_list
