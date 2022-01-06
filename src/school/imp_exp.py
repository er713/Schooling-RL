import csv
from typing import List
from os.path import isfile

from . import Result


def import_results(path: str) -> List[Result]:
    """
    Function for importing results
    :param path: Path to csv file
    :return: List of imported Results
    """
    results = []
    with open(path, 'r') as file:
        reader = csv.DictReader(file, dialect='unix', fieldnames=None)
        for row in reader:
            results.append(Result.create_from_dict(row))
    return results


def export_results(path: str, results: List[Result]) -> None:
    """
    Function for exporting results
    :param path: Path to csv file
    :param results: List with Results to export.
    """
    writeHeader = not isfile(path)

    with open(path, 'a') as file_csv:
        writer = csv.DictWriter(file_csv, dialect='unix', fieldnames=list(results[0].__dict__.keys()))
        if writeHeader:
            writer.writeheader()
        writer.writerows([res.get_dict() for res in results])


def export_improvements(path: str, improvements: List[dict]):
    writeHeader = not isfile(path)

    with open(path, 'a') as file_csv:
        writer = csv.DictWriter(file_csv, dialect='unix', fieldnames=improvements[0].keys(),
                                quoting=csv.QUOTE_NONE, escapechar=' ')
        if writeHeader:
            writer.writeheader()
        writer.writerows(improvements)
