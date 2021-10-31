from typing import List, Optional
import matplotlib.pyplot as plt

from . import import_results, Result


class Plotter:
    """
    Class responsible for draw results plots 
    """

    def __init__(self) -> None:
        self.results = []

    def plot_learning_results(self, result: float) -> None:
        """
        Function responsible for plot results (student's mean score on exams (epochs?)) during learning
        :param result: student's mean score on exam (epoch)
        """
        self.results.append(result)
        self.__plot(self.results)

    def plot_learning_save(self, path: str) -> None:
        """
        Saving plot created during learning
        """
        self.__plot(self.results, path)

    @staticmethod
    def plot_results(results: List[float], savePath: str = None) -> None:  # zwykły plot gdyby ktoś chciał skorzystać
        Plotter.__plot(results, savePath)

    @staticmethod
    def plot_from_csv(path: str, savePath: str = None):
        """
        Function responsible for plot results from .csv file
        """
        results = import_results(path)
        Plotter.plot_results(Result.get_exams_means(results), savePath)

    @staticmethod
    def __plot(results: List[float] = [], savePath: str = None):
        fig = plt.figure()
        plt.plot(range(len(results)), results)
        plt.ylim([0., 1.])
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik studentów na egzaminie')
        plt.title('Wykres zdobytych nagród w czasie')
        plt.show()
        if savePath is not None:
            fig.savefig(savePath, dpi=fig.dpi)
