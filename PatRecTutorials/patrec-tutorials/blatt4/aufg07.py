import matplotlib.pyplot as plt
from sklearn import svm
from common.classification import ClassificationEvaluator, CrossValidation
from common.data_provider import DataProvider
from common import visualization
import numpy as np


def aufg07():
    #
    # Im Gegensatz zu den bisher verwendeten Klassifikatoren, die Klassengebiete
    # explizit ueber (Mischungen von) Gaussverteilungen modellieren, sollen nun
    # diskriminative Klassifikatoren untersucht werden. Als besonders leistungs-
    # faehig haben sich die in der Vorlesung behandelten Support Vector Maschinen
    # erwiesen.
    # Anhand der beiden bisher benutzten aber auch unter Verwendung eigener
    # Datensaetze soll nun die Mustererkennung mittels Support Vector Machines
    # untersucht werden.
    # Fuer Python stellt die Bibliothek 'sklearn' die Implementierung einer SVM
    # bereit.
    # http://scikit-learn.org/stable/modules/svm.html#svm
    # Zur Visualisierung der SVM kann die Funktion visualization.plot_svm
    # genutzt werden.

    # Der zweidimensionale Datensatz data2d enthaelt 3 Klassen.
    # Trainieren Sie zunaechst eine lineare Support Vector Maschine, die zur
    # Trennung von Klasse 0 und 2 verwendet werden soll. Klasse 1 wird hier
    # nicht betrachtet, da sie sich nicht linear von den Klassen 0 und 2 trennen
    # laesst (Siehe unten).
    # Wie hoch ist der Klassifikationsfehler im Vergleich zum Normalverteilungs-
    # klassifikator? Visualisieren Sie die resultierende Trennfunktion. Im Modul
    # visualization steht dafuer die Methode plot_svm bereit.
    # Diskutieren Sie den Einfluss der Slack-Variablen (in Form des C-Parameters)
    # auf die Trennfunktion und damit den entstehenden Klassifikationsfehler.
    #
    # Fuer die Evaluierung koennen Sie im Folgenden wieder den ClassificationEvaluator
    # aus dem Modul common.classifiers verwenden.


    train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
    test_data_provider = DataProvider(DataProvider.DATA2DROOT_TEST)

    train_data, train_labels = train_data_provider.get_dataset_and_labels()
    test_data, test_labels_gt = test_data_provider.get_dataset_and_labels()

    train_labels = train_labels.astype(dtype='float64')
    test_labels_gt = test_labels_gt.astype(dtype='float64')

    train_data_02 = np.concatenate((train_data_provider.get_class_arr(0), train_data_provider.get_class_arr(2)))
    train_labels_02 = np.concatenate((train_labels[train_labels==0], train_labels[train_labels==2]))

    test_data_02 = np.concatenate((test_data_provider.get_class_arr(0), test_data_provider.get_class_arr(2)))
    test_labels_02 = np.concatenate((test_labels_gt[test_labels_gt==0], test_labels_gt[test_labels_gt==2]))

    clf = svm.LinearSVC()
    clf.fit(train_data_02, train_labels_02)

    estimated_labels = clf.predict(test_data_02)

    evals = ClassificationEvaluator(estimated_labels, test_labels_02)
    error_rate, n_wrong, n_samples = evals.error_rate()
    print(error_rate, n_wrong, n_samples)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    visualization.plot_svm(ax, test_data_02, test_labels_02, clf, step_size=0.1)
    #plt.show()

    # raise NotImplementedError('Implement me')

    # Trainieren Sie nun eine SVM fuer die Klassen 1 und 2.
    # Evaluieren Sie, welcher Kernel geeignet ist, um das Problem zu loesen.

    train_data_12 = np.concatenate((train_data_provider.get_class_arr(1), train_data_provider.get_class_arr(2)))
    train_labels_12 = np.concatenate((train_labels[train_labels==1], train_labels[train_labels==2]))

    test_data_12 = np.concatenate((test_data_provider.get_class_arr(1), test_data_provider.get_class_arr(2)))
    test_labels_12 = np.concatenate((test_labels_gt[test_labels_gt==1], test_labels_gt[test_labels_gt==2]))

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    for kernel in kernels:
        clf = svm.SVC(kernel=kernel)
        clf.fit(train_data_12, train_labels_12)

        estimated_labels = clf.predict(test_data_12)

        evals = ClassificationEvaluator(estimated_labels, test_labels_12)
        error_rate, n_wrong, n_samples = evals.error_rate()
        print(kernel, error_rate, n_wrong, n_samples)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        visualization.plot_svm(ax, test_data_12, test_labels_12, clf, step_size=0.1)
        #plt.show()

    # raise NotImplementedError('Implement me')

    # Trainieren Sie nun eine Multi-Class SVM zur Loesung des 3-Klassenproblems
    # unter Verwendung eines geeigneten Kernels.
    # Wie koennen die optimalen kernelspezifischen Parameter sinnvoll ermittelt
    # werden?
    # Hinweis: Starten Sie zunaechst mit den Grundeinstellungen der Bibliothek.

    clf = svm.SVC()
    clf.fit(train_data, train_labels)

    estimated_labels = clf.predict(test_data)

    evals = ClassificationEvaluator(estimated_labels, test_labels_gt)
    error_rate, n_wrong, n_samples = evals.error_rate()
    print('3 Klassen: bcf', error_rate, n_wrong, n_samples)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    visualization.plot_svm(ax, test_data, test_labels_gt, clf, step_size=0.1)
    # plt.show()


    #for i in range(1, 100):

    #    clf = svm.SVC(kernel = 'rbf')
    #    clf.fit(train_data, train_labels)
    #    estimated_labels = clf.predict(test_data)
    #    
    #    evals = ClassificationEvaluator(estimated_labels, test_labels_gt)
    #    error_rate, n_wrong, n_samples = evals.error_rate()
    #    print('gamma = ', i, ' : ', error_rate, n_wrong, n_samples)

    #raise NotImplementedError('Implement me')

    # Vergleichen Sie Ihre Ergebnisse mit den bisher erzielten
    # Klassifikationsfehlerraten.


if __name__ == '__main__':
    aufg07()
