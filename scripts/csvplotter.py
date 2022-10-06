import csv
import enum
import numpy as np
import matplotlib.pyplot as plt

class Func:
    def __init__(self, xExpr, yExpr, errorExpr=None, label=None):
        self.xExpr = xExpr
        self.yExpr = yExpr
        self.errorExpr = errorExpr
        self.hasError = self.errorExpr is not None
        self.label = label

    def x(self, args):
        return eval(self.xExpr, args)
    
    def y(self, args):
        return eval(self.yExpr, args)
    
    def error(self, args):
        return eval(self.errorExpr, args)


def main(filePath, logX, logY, xLabel, yLabel, title, showLegend, fontSize, funcs):
    xData = [[]] * len(funcs)
    yData = [[]] * len(funcs)
    errorData = [[]] * len(funcs)

    with open(filePath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row = {k : float(v) for k, v in row.items()}
            
            for i, func in enumerate(funcs):
                xData[i].append(func.x(row))
                yData[i].append(func.y(row))

                if func.hasError:
                    errorData[i].append(func.error(row))

    if fontSize is not None:
        plt.rc('font', size=fontSize)
    
    for i in range(len(funcs)):
        plt.scatter(xData[ i ], yData[ i ], label=func.label)
    
        if funcs[ i ].hasError:
            plt.errorbar(xData[ i ], yData[ i ], yerr=errorData[ i ], fmt='o')
    
    if showLegend:
        plt.legend()

    if logX:
        plt.semilogx()
    if logY:
        plt.semilogy()

    if xLabel is not None:
        plt.xlabel(xLabel)

    if yLabel is not None:
        plt.ylabel(yLabel)

    if title is not None:
        plt.title(title)

    plt.show()


# main('data.txt', False, True, 'Number of basis functions', 'Error (Ht)', None, False, 20, [Func('nBasis', 'error', 'stddev')])
# main('data.txt', False, False, 'Number of basis functions', 'Running time (s)', None, False, 20, [Func('nBasis', 'time / 1000')])
main('qmcData.txt', True, True, 'Number of function evaluations', 'Error (Ht)', None, False, 20, [Func('evaluations', 'error', 'stddev')])
