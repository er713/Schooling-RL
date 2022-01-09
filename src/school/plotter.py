from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import copy
from . import import_results, Result
import numpy as np
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#330000", "#333300", "#003300",
    "#003333", "#000033", "#330033", "#000000", "#663300", "#336600", "#006633",
    "#003366", "#330066", "#660033", "#990000", "#999900", "#009900", "#009999",
    "#000099", "#990099", "#404040", "#CC6600", "#66CC00", "#00CC66", "#0066CC",
    "#6600CC", "#606060", "#FF8000", "#80FF00", "#00FF80", "#0080FF", "#7F00FF",
    "#FF007F", "#FF3333", "#FFFF33", "#33FF33", "#33FFFF", "#3333FF", "#FF33FF",
    "#A0A0A0", "#FFB266", "#B2FF66", "#66FFB2", "#66B2FF", "#B266FF", "#FF66B2",
    "#FF9999", "#FFFF99", "#99FF99", "#99FFFF", "#9999FF", "#FF99FF", "#E0E0E0", 
    "#FFE5CC","#E5FFCC",  "#CCFFE5", "#CCE5FF", "#FFCCFF"]) 
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
            results.append(Result.get_exams_means(import_results(file)))
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
    def plot_from_csv_with_std(files: List[str], save_path: str, title: str):
        results = []
        for file in files:
            results.append(Result.get_exams_means(import_results(file)))
        m = np.mean(results,axis=1)
        std = np.std(results,axis=1)
        fig = plt.figure()
        plt.plot(m)
        plt.fill_between(range(len(m)),m+std,m-std, alpha=.2)
        plt.xlabel("Numer egzaminu")
        plt.ylabel("Wynik")
        plt.title(title)
        plt.legend()
        plt.show()
        fig.savefig(save_path, dpi=fig.dpi)
        plt.close()

    @staticmethod
    def draw_tasks_distribution(path):
        df = pd.read_csv(path,header=None,skiprows=[0])
        headers = [(x//7, (x%7)-3)  for x in range(int(df.shape[1])) ]
        # df.columns = headers
        fig = plt.figure()
        plt.plot(df, label=headers)
        plt.title('Dystrubucja zadań')
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Ile razy zadanie zostało wybrane')
        plt.legend()
        plt.show()
        fig.savefig(path.replace('.csv',''), dpi=fig.dpi)
        plt.close()

        

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
        """
        Function responsible for save results to csv file
        :param: str save_path: Path where user want save csv file
        """
        np.savetxt(save_path,self.results)

    @staticmethod
    def draw_student_progress(path):
        df = pd.read_csv(path,header=None,skiprows=[0])
        
        fig = plt.figure()
        plt.plot(df, label=df.columns )
        plt.plot(np.mean(df,axis=1),label='Średnia zmiana')
        plt.title('Progres studentw')
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Zmiana poziomu')
        plt.legend()
        plt.show()
        fig.savefig(path.replace('.csv',''), dpi=fig.dpi)
        plt.close()  
