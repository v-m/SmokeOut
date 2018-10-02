#
# Author: Vincenzo Musco (http://www.vmusco.com)
from config import DATASET_ROOT_ALL
from tools.static import datasetOutFile, runForNr

import sys
import os
import csv
import glob

class ExcelFileWritingHelper:
    def __init__(self, workbook):
        self.workbook = workbook
        self.excelstyles = {}

    def newsheet(self, name):
        self.worksheet = self.workbook.add_worksheet(name)
        self.headerinfo = []
        self.lines = [[]]
        self.linesExcelStyles = [[]]
        self.cLine = 0
        # self.cCol = 0

    def newLine(self):
        '''
        This function create a new line in the document.
        It is responsible of persisting in the Excel file the last line as well as the header if it's the first line!
        :return: The produced text header to show on console if needed (cf. printLine(e)).
        '''
        producedHeader = None

        if len(self.lines) == 1 and len(self.headerinfo) > 0:
            #Prine header?
            producedHeader = []
            nblines = max(list(map(lambda x: len(x), self.headerinfo)))

            for l in range(nblines):
                producedHeader.append([])
                for r in range(len(self.headerinfo)):
                    if len(self.headerinfo[r]) > 0 and len(self.headerinfo[r])-1 >= l:
                        producedHeader[-1].append(self.headerinfo[r][l])
                        self.worksheet.write(self.cLine, r, self.headerinfo[r][l])
                    else:
                        producedHeader[-1].append("")

                self.cLine += 1

        #Print line
        for i in range(len(self.lines[-1])):
            style = self.linesExcelStyles[-1][i]
            if style is not None:
                self.worksheet.write(self.cLine, i, self.lines[-1][i], style)
            else:
                self.worksheet.write(self.cLine, i, self.lines[-1][i])

        self.cLine += 1

        # self.cCol = 0
        self.lines.append([])
        self.linesExcelStyles.append([])

        self.headerinfos = None
        return producedHeader

    def write(self, value=None, headerinfos = None, fgcolor = None, bgcolor = None):
        '''
        Write the next column in the current line (as well as the header information if it's the first line)
        :param value: the next value
        :param headerinfos: the related column lines for this column (can be None for no header)
        :return:
        '''
        stylekey = "{}{}".format(0 if fgcolor is None else fgcolor, 0 if bgcolor is None else bgcolor)
        style = None
        if stylekey != "00":
            if stylekey not in self.excelstyles:
                style = self.workbook.add_format()
                self.excelstyles[stylekey] = style

                if fgcolor is not None:
                    style.set_font_color(fgcolor)
                if bgcolor is not None:
                    style.set_bg_color(bgcolor)
            else:
                style = self.excelstyles[stylekey]

        if headerinfos is not None:
            self.headerinfo.append(headerinfos)
        else:
            self.headerinfo.append([])

        if value is not None:
            self.lines[-1].append(value)
        else:
            self.lines[-1].append("")

        self.linesExcelStyles[-1].append(style)

    def resetLine(self):
        self.lines[-1] = []
        self.linesExcelStyles[-1] = []

    @staticmethod
    def printLine(e, where=sys.stdout, sep='\t'):
        '''
        This function writes the produced line with the specific separator
        By default writes with tabs separator to stdout.
        :param e: the ExcelFileWritingHelper object from which the next line should be printed
        :param where: Where lines should be written
        :param sep: separator for lines
        :return:
        '''
        firstLine = e.newLine()

        if firstLine is not None:
            for line in firstLine:
                print(sep.join(line), file=where)

        print(sep.join(list(map(lambda x: str(x), e.lines[-2]))), file=where)

    @staticmethod
    def computeExcelLetter(v):
        rest = int(v / 26)
        curchar = chr(ord('A') + (int(v) % 26))

        if rest > 0:
            return '{}{}'.format(ExcelFileWritingHelper.computeExcelLetter(rest - 1), curchar)

        return curchar

    def currentLine(self):
        return len(self.lines)


def readDatasetCsvFile(datasetName, algoName, runinfo=None):
    outFile = datasetOutFile(datasetName, algoName, runinfo=runinfo)

    if os.path.exists(outFile):
        return readDatasetCsvFileLogic(outFile)

    return None

def readDatasetCsvFileLogic(outFile):
    with open(outFile, 'r') as csvfile:
        return [l for l in csv.reader(csvfile) if l[0] != ""]

def listRunsFromEverySources(datasetName, algoName, runinfo=None, sources=DATASET_ROOT_ALL, wildcardmask='*'):
    files = set()
    for source in sources:
        outFile = datasetOutFile(datasetName, algoName, runinfo=runForNr(runinfo, wildcardmask), datasetroot=source)
        [files.add(f) for f in glob.glob(outFile)]

    return list(files)