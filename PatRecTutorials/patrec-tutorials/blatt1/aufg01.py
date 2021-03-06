from common.data_provider import DataProvider
import matplotlib.pyplot as plt
from common.classification import KNNClassifier
from common.classification import ClassificationEvaluator
from common.classification import CrossValidation

def aufg01():
    # Zur Einfuehrung werden auf einem Beispieldatensatz Klassifikatoren
    # implementiert. Der Datensatz data2d enthaelt zweidimensionalen
    # Trainingsmerkmale fuer drei Musterklassen. Fuer diese Daten soll eine
    # Klassifikation ueber Naechster-Nachbar realisiert  werden.
    # Achtung: Gestalten Sie Ihre Implementierung so, dass Sie die Klassifikatoren
    # fuer zukuenftige Aufgaben wiederverwenden koennen.

    # Im Folgenden wird der Beispieldatensatz ueber die Klasse DataProvided
    # geladen und anschliessend visualisiert. Machen Sie sich mit sowohl dem
    # Laden als auch der Visualisierung vertraut, da Sie in den kommenden
    # Aufgaben diese Aspekte wiederverwenden werden.
    # http://matplotlib.org/users/pyplot_tutorial.html
    #
    # Nuetzliche Funktionen: plt.scatter
    # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter
    #
    # Tipp: zu einer besseren Visualisierung sollten alle scatter plots in Matplotlib
    # immer mit dem Argument "edgecolor=(0, 0, 0)" aufgerufen werden.

    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # fig, (ax, ax2) = plt.subplots(1, 2)

    data = train_data_provider.get_class_arr(0)
    ax.scatter(data[:, 0], data[:, 1], c='#FF0000', edgecolor=(0, 0, 0))
    data = train_data_provider.get_class_arr(1)
    ax.scatter(data[:, 0], data[:, 1], c='#00FF00', edgecolor=(0, 0, 0))
    data = train_data_provider.get_class_arr(2)
    ax.scatter(data[:, 0], data[:, 1], c='#0000FF', edgecolor=(0, 0, 0))

    # plt.show()

    #
    # Implementieren Sie einen Naechster-Nachbar-Klassifikator.
    # Vervollstaendigen Sie dazu die Klasse KNNClassifier im Modul common.classifiers.
    # Testen Sie verschiedene Abstandsmasse. Welche halten Sie insbesondere fuer sinnvoll?
    train_data, train_labels = train_data_provider.get_dataset_and_labels()
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()
    test_labels_gt = test_labels_gt.astype('float64')
    train_labels = train_labels.astype('float64')

    knn_classifier = KNNClassifier(k_neighbors=70, metric='cityblock') # euclidean, cityblock, chebyshev
    knn_classifier.estimate(train_data, train_labels)
    estimated_labels = knn_classifier.classify(test_data)

    # print(len(estimated_labels==0))

    # print(test_labels_gt.shape, estimated_labels.shape)
    # print(test_labels_gt.dtype, estimated_labels.dtype)

#
    #data_x = test_data[mask, 0]
    #data_y = test_data[mask, 1]
    #ax2.scatter(data_x, data_y, c='#FF0000', edgecolor=(0, 0, 0))
    #data = test_data[estimated_labels == 1]
    #ax2.scatter(data[:, 0], data[:, 1], c='#00FF00', edgecolor=(0, 0, 0))
    #data = test_data[estimated_labels == 2]
    #ax2.scatter(data[:, 0], data[:, 1], c='#0000FF', edgecolor=(0, 0, 0))
    #fig.tight_layout()
    #plt.show()

    #
    # Nutzen Sie zur Evaluation der Ergebnisse die Klasse ClassificationEvaluator
    # im Modul common.classifiers.

    evals = ClassificationEvaluator(estimated_labels, test_labels_gt)

    error_rate, n_wrong, n_samples = evals.error_rate()

    print(error_rate, n_wrong, n_samples)

    # raise NotImplementedError('Implement me')

    # Ein NN-Klassifikator alleine ist meist nicht ausreichend. Erweitern Sie
    # den Klassifikator zum k-NN Klassifikator.
    # Fuer den Mehrheitsentscheid ist das defaultdict nuetzlich (siehe intro).
    # https://docs.python.org/3/library/collections.html#collections.defaultdict

    # Trainingsparameter sollten immer per Kreuzvalidierung auf den Trainingsdaten
    # optimiert werden. Mit den besten Parametern wird dann ein Klassifikator
    # erstellt und auf den Testdaten evaluiert.
    # Nutzen Sie die Klasse CrossValidation im Modul classification um den
    # Parameter k zu optimieren.
    # In den folgenden Aufgaben ist es Ihnen freigestellt, ob Sie Kreuzvalidierung
    # nutzen oder direkt auf den Testdaten optimieren.
    
    cross = CrossValidation(train_data, train_labels, 5)

    for i in range(65, 76):

        knn = KNNClassifier(i, 'cityblock')
        crossval_overall_result, crossval_class_results = cross.validate(knn)
        print('Anzahl der Nachbarn = ', i, ' : ', crossval_overall_result)

    ''' Optimum bei k_neighbours = 70 '''

    # raise NotImplementedError('Implement me')




if __name__ == '__main__':
    aufg01()
