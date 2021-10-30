from typing import List, Optional
import matplotlib.pyplot as plt


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
       

    def plot_results(self,results : List[float]) -> None: #zwykły plot gdyby ktoś chciał skorzystać
        self.__plot(results)
    
        
    def plot_from_csv(self,path : str):
        """
        Function responsible for plot results from .csv file
        """
        raise NotImplementedError()
    
    
    def __plot(self,results : List[float] = []):
        fig = plt.figure()
        plt.plot(range(len(results)),results)
        plt.xlabel('Numer egzaminu')
        plt.ylabel('Średni wynik studentów na egzaminie')
        plt.title('Wykres zdobytych nagród w czasie')
        plt.show()
        
        #fig.savefig('temp.png', dpi=fig.dpi)
    