import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

CHART_TITLE = "MSE"
CHART_TITLE_COLOR = "#f0bd18"
CHART_TITLE_FONT_SIZE = 12

CHART_XLABEL = "# Epoch"
CHART_YLABEL = "Loss"
CHART_AXIS_LBLS_FONT_SIZE = 10


#Plot class to represent simple chart in PYQT5 app
class Canvas(FigureCanvas):

    def __init__(self, parent, width=5, height=4, dpi=80):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)

        self.set_chart_titles()
        self.setParent(parent)

        self.draw()

    #Method sets chart titles and properties
    def set_chart_titles(self):
        self.ax.set_title(CHART_TITLE)
        self.ax.title.set_color(CHART_TITLE_COLOR)

        self.ax.set_xlabel(CHART_XLABEL)
        self.ax.set_ylabel(CHART_YLABEL)

        self.ax.xaxis.label.set_size(CHART_AXIS_LBLS_FONT_SIZE)
        self.ax.title.set_size(CHART_TITLE_FONT_SIZE)