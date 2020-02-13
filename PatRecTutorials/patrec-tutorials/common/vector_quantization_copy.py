import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.metrics import mean_squared_error


class Lloyd(object):

    def cluster(self, samples, codebook_size, prune_codebook=False):
        """Partitioniert Beispieldaten in gegebene Anzahl von Clustern.

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
            codebook_size: Anzahl von Komponenten im Codebuch
            prune_codebook: Boolsches Flag, welches angibt, ob das Codebuch
                bereinigt werden soll. Die Bereinigung erfolgt auf Grundlage
                einer Heuristik, die die Anzahl der, der Cluster Komponente
                zugewiesenen, Beispieldaten beruecksichtigt.
                Optional, default=False

        Returns:
            codebook: ndarry mit codebook_size Codebuch Vektoren,
                zeilenweise, shape=(codebook_size,t)

        mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
        """
        #
        # Bestimmen Sie in jeder Iteration den Quantisierungsfehler und brechen Sie
        # das iterative Verfahren ab, wenn der Quantisierungsfehler konvergiert
        # (die Aenderung einen sehr kleinen Schwellwert unterschreitet).
        # Nuetzliche Funktionen: scipy.distance.cdist, np.mean, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Fuer die Initialisierung mit zufaelligen Punkten: np.random.permutation
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html

        # raise NotImplementedError('Implement me')

        codebook = np.zeros((codebook_size, 2))
        
        np.random.seed(42)


        for i in range(codebook_size):
                x = np.random.random_integers(0, len(samples))
                codebook[i, :] = samples[x, :]
        
        err = 0
        xx = 1

        distance = cdist(samples, codebook)
        labels = distance.argmin(axis=1)     
        
        for i in range(len(samples[:,0])):
            key = distance[i, :].argmin()
            err += distance[i, key]
            # clusters[str(key)].append(samples[i,:])

    
        it = 0
        while xx > 1e-7:
            codebook_neu = np.zeros(codebook.shape)
            new_err = 0

            for i in range(codebook_size):
                class_data = samples[labels == i]
                x = np.mean(class_data, axis=0)
                codebook_neu[i, :] = x


            #for i in range(codebook_size):
            #    item = np.asarray(clusters[str(i)])
            #    mean = (np.mean(item[:, 0]), np.mean(item[:, 1]))
            #    codebook_neu[i, :] = mean

            distance = cdist(samples, codebook_neu)
            labels = distance.argmin(axis=1) 
            
            for i in range(len(samples[:,0])):
                key = distance[i, :].argmin()
                new_err += distance[i, key]
            
            xx = np.abs(err-new_err)/new_err
            print(it)
            err = new_err
            codebook = codebook_neu 
            it+=1

        return codebook

             