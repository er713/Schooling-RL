from typing import List, Optional
import matplotlib.pyplot as plt
import os
import re
import copy
from . import import_results, Result


class Plotter:
    """
    Class responsible for draw and save results plots 
    """

    def __init__(self) -> None:
        self.results = []

    def plot_learning_results(self, result: float) -> None:
        """
        Function responsible for plot results (student's mean score on exams (epochs?)) during learning
        :param result: student's mean score on exam (epoch)
        """
        self.results.append(result)
        self.__plot_dynamically(self.results)

    def plot_learning_save(self, save_path: str) -> None:
        """
        Saving plot created during learning
        :param str save_path: Path where user want save plot.
        """
        self.__plot_dynamically(self.results, save_path)

    @staticmethod
    def plot_results(results: List[float], save_path: str = None) -> None:
        """
        Function responsible for plot results as list
        :param List[float] results: list contains results
        :param str save_path: Path where user want save plot.
        """
        Plotter.__plot(results, save_path)

    @staticmethod
    def plot_from_csv(path: str, save_path: str = None):
        """
        Function responsible for plot results from .csv file
        :param str save_path: Path where user want save plot.
        """
        results = import_results(path)
        Plotter.plot_results(Result.get_exams_means(results), save_path)

    @staticmethod
    def plot_multiple_from_csv(files: List[str], save_path: str = None, nSkills: int = None):
        """
        Function responsible for plot results from multiple .csv files on one plot
        :param files: paths to .csv files
        :param str save_path: Path where user want save plot.
        """
        results = []
        labels = []
        for file in files:
            file_path = copy.deepcopy(file)
            file = file.replace("./data/", "", 1)

            teacher = re.match("^[a-zA-Z]+", file)
            file = file.replace(f"{teacher[0]}/", "", 1)

            student = re.match("^[a-zA-Z]+", file)
            file = file.replace(f"{student[0]}__", "", 1)

            nStudents = re.match("^[0-9]+", file)
            file = file.replace(f"{nStudents[0]}_", "", 1)

            skills = re.match("^[0-9]+", file)
            file = file.replace(f"{skills[0]}__", "", 1)

            tasks = re.match("^[0-9]+", file)
            if nSkills == int(skills[0]):
                labels.append(f"{teacher[0]}-{skills[0]}-{int(int(tasks[0])/int(skills[0]))}")
                results.append(Result.get_exams_means(import_results(file_path)))

        Plotter.__plot_multiple(results, save_path, labels)



    @staticmethod
    def plot_compare_results(results_path: List[str],
                             save_path: str = None,
                             custom_titles: List[str] = None):
        """
        Function responsible for plot multiple results (suplots)
        :param List[str] results_path: List contains results .csv files paths.
        :param str save_path: Path where user want save plot.
        :param List[str] custom_titles: List contains custom subplots titles.
            Defaults subplots titles are .csv files names.
            
            os.path.basename(your_path)
        """
        titles = custom_titles if custom_titles else \
            [os.path.basename(path) for path in
             results_path]  # Uwaga może nie działać prawidłowo na Windowsie -- potrzebny feedback xD
        n_row = len(results_path)
        fig = plt.figure()

        for i in range(n_row):
            results = Result.get_exams_means(import_results(results_path[i]))
            Plotter.__subplots(n_row, i + 1, results, titles[i])

        plt.tight_layout()
        plt.show()

        if save_path:
            fig.savefig(save_path, dpi=fig.dpi)


    @staticmethod
    def plot_compare_results(results_path: List[str],
                             save_path: str = None,
                             custom_titles: List[str] = None):
        """
        Function responsible for plot multiple results (suplots)
        :param List[str] results_path: List contains results .csv files paths.
        :param str save_path: Path where user want save plot.
        :param List[str] custom_titles: List contains custom subplots titles.
            Defaults subplots titles are .csv files names.
            
            os.path.basename(your_path)
        """
        titles = custom_titles if custom_titles else \
            [os.path.basename(path) for path in
             results_path]  # Uwaga może nie działać prawidłowo na Windowsie -- potrzebny feedback xD
        n_row = len(results_path)
        fig = plt.figure()

        for i in range(n_row):
            results = Result.get_exams_means(import_results(results_path[i]))
            Plotter.__subplots(n_row, i + 1, results, titles[i])

        plt.tight_layout()
        plt.show()

        if save_path:
            fig.savefig(save_path, dpi=fig.dpi)

    @staticmethod
    def __subplots(n_rows: int, i_row: int, results: List[float], title: str):
        plt.subplot(n_rows, 1, i_row)
        plt.plot(range(len(results)), results)
        plt.ylim([0., 1.])
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik')
        plt.title(title)

    @staticmethod
    def __plot_dynamically(results: List[float] = [], save_path: str = None):
        plt.ion()
        ax = plt.gca()
        ax.set_autoscale_on(True)
        line, = ax.plot(range(len(results)), results)
        line.set_ydata(results)
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.set_xlabel('Numer egzaminu')
        ax.set_ylabel('Średni wynik studentów na egzaminie')
        ax.set_title('Wykres zdobytych nagród w czasie')
        ax.set_ylim([0, 1])
        plt.draw()
        plt.pause(1)

        if save_path:
            ax.figure.savefig(save_path, dpi=ax.figure.dpi)

    @staticmethod
    def __subplots(n_rows: int, i_row: int, results: List[float], title: str):
        plt.subplot(n_rows, 1, i_row)
        plt.plot(range(len(results)), results)
        plt.ylim([0., 1.])
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik')
        plt.title(title)

    @staticmethod
    def __plot_dynamically(results: List[float] = [], save_path: str = None):
        plt.ion()
        ax = plt.gca()
        ax.set_autoscale_on(True)
        line, = ax.plot(range(len(results)), results)
        line.set_ydata(results)
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.set_xlabel('Numer egzaminu')
        ax.set_ylabel('Średni wynik studentów na egzaminie')
        ax.set_title('Wykres zdobytych nagród w czasie')
        ax.set_ylim([0, 1])
        plt.draw()
        plt.pause(1)

        if save_path:
            ax.figure.savefig(save_path, dpi=ax.figure.dpi)

    @staticmethod
    def __plot(results: List[float] = [], save_path: str = None):
        fig = plt.figure()
        plt.plot(range(len(results)), results)
        plt.ylim([0., 1.])
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik studentów na egzaminie')
        plt.title('Wykres zdobytych nagród w czasie')
        plt.show()

        if save_path:
            fig.savefig(save_path, dpi=fig.dpi)

    @staticmethod
    def __plot_multiple(results: List[List[float]] = [[]], save_path: str = None, labels: List[str] = None):

        fig = plt.figure()
        for result, label in zip(results,labels):
            plt.plot(range(len(result)), result, label=label)

        plt.ylim([0., 1.])
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik studentów na egzaminie')
        plt.title('Wykres zdobytych nagród w czasie')
        plt.legend()
        plt.show()

        if save_path:
            fig.savefig(save_path, dpi=fig.dpi)
        plt.close()
    
    def dump_results_to_csv(self,
                        save_path: str = None) -> None:
        np.savetxt(save_path,self.results)
        
