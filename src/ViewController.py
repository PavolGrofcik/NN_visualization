import sys, random, functools
import matplotlib as mp

from matplotlib import colors
from matplotlib import cm as cmap

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from ANN.FigureCanvas import Canvas
from ANN.model import NN_regression

#####################################################################
APP_NAME = "NN visualization"
APP_STYLE = "Fusion"

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 800
WINDOW_LEFT_POS = 250
WINDOW_TOP_POS = 100

WINDOW_ICO = "Logo/NN_brain.png"
WINDOW_BACK_COLOR = "#FFFFFF"

#Pixels size for NN topology
NEURON_SIZE = 50
NEURON_Y_SPACE = 60
NEURON_TOP_OFFSET = 50
NEURON_X_SPACE = 300
NEURON_LEFT_POS = 100
NEURON_TOP_POS = 200

NEURON_PEN_WIDTH = 4
NEURON_PEN_COLOR = Qt.darkBlue
NEURON_PEN_STYlE = Qt.SolidLine

WEIGHT_PEN_WIDTH = 2.5
WEIGHT_PEN_STYLE = Qt.SolidLine
WEIGHT_PEN_COLOR = QColor("#f0f0f0") #ebebeb
WEIGHT_OUTPUT_PEN_COLOR = QColor("#1e74d6")

#NN Architecture parameters
NEURON_LAYERS = 3
NEURONS = [3, 5, 1]
LEARNING_RATE = 0.0007
SEED = 1
NN_SAMPLES = 2
EPOCHS = 25

TITLE_FONT = "Calibri Light"
TITLE_FONT_SIZE = 17
TITLE_FONT_COLOR = "#f0bd18"            #f0bd18
TITLE_MAIN = "NN vizualizácia učenia"
TITLE_LAYERS_FONT_COLOR = '#d15213'     #d15213
TITLE_LAYERS_FONT_SIZE = 14
TITLE_LAYERS_FONT = "Calibri Light"

#NN dots labels props
LBL_DOTS_PLACEHOLDER = ""
LBL_DOTS_FONT = "Calibri Light"
LBL_DOTS_FONT_SIZE = 9
LBL_DOTS_FONT_COLOR = "#db1818" # "1e74d6"
LBL_DOTS_INPUT_COLOR = "#d15213" #f0bd18
LBL_DOTS_WIDTH = 30
LBL_DOTS_HEIGHT = 15

LBL_WEIGHT_FONT = "Calibri Light"
LBL_WEIGHT_FONT_SIZE = 10
LBL_WEIGHT_FONT_COLOR = "#d15213"
LBL_WEIGHT_PLACEHOLDER = ""
LBL_TRANSPARENT_BACKGROUND = "rgba(0,0,0,0%)"

LBL_INFO_FONT = "Calibri"
LBL_INFO_FONT_SIZE = 14
LBL_INFO_FONT_COLOR = "#1e74d6"
BUTTON_FONT_COLOR = "#d15213"

LABEL_CACHE_SIZE = (1000, 800)
PIXMAP_SIZE = (1000, 800)
PIXMAP_BACKGROUND_COLOR = Qt.white
CANVAS_POSITION = (1150, 50)

COLOR_PALLETE = "viridis"        #plasma, cividis, viridis...

STATUS_LBLS = ["Forward propagation", "Error calculation", "Backpropagation"]
STATUS_ARROWS = [
    r"C:\Users\grofc\Desktop\Projekt Ing\NN_visualization\ANN\Logo\arrow_forward.png",
    r"C:\Users\grofc\Desktop\Projekt Ing\NN_visualization\ANN\Logo\arrow_backward.png"]


class ViewController(QMainWindow):
    def __init__(self):
        super().__init__()

        #Set a cache for drawing in order to avoid paintEvent
        self.cache = QLabel(self)
        self.cache.resize(*LABEL_CACHE_SIZE)
        self.cache.move(0, 0)

        self.pixmap = QPixmap(*PIXMAP_SIZE)
        self.pixmap.fill(QColor(PIXMAP_BACKGROUND_COLOR))

        self.init_app_window()
        self.init_draw_flags()
        self.init_canvas()
        self.init_NN()
        self.draw_NN_architecture()

        #self.update_plot()
        self.show()

    def init_app_window(self):
        "Method initializes window properties"
        self.setGeometry(WINDOW_LEFT_POS, WINDOW_TOP_POS,
                         WINDOW_WIDTH,
                         WINDOW_HEIGHT)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(WINDOW_ICO))
        self.setFixedSize(self.size())
        self.setStyleSheet("background-color: " +
                           WINDOW_BACK_COLOR)  # Set background window to white

    def init_draw_flags(self):
        "Method initializes flags for paint_event method"
        self.epoch_counter = 0    #Initial drawing of plot
        self.reseted = False
        self.NN_loaded = False
        self.timer = None

        self.weight_lines = {}  #Dict of weight lines
        self.titles = {}        #Dict of titles
        self.lbl_dots = {}      #Dict of dots products
        self.weight_lbls = {}   #Dict of weights
        self.neurons_coords = []

        #Calculates neuron topology
        self.get_NN_topology_size()
        self.get_NN_synapses_size()

        self.add_buttons()          #Add butons to window
        self.add_titles()           #Add titles to window
        self.add_neuron_labels()    #Add neuron labels to window
        self.add_weight_labels()    #Add weight labels to window

    def init_canvas(self):
        "Method initializes canvas for plotting"
        self.chart = Canvas(self)
        self.chart.move(*CANVAS_POSITION)

    #Method initializes Neural Network
    def init_NN(self):
        self.NN = NN_regression.NeuralNetwork(X_train=None, Y_train=None,
                                              layers=NEURONS, learn_rate=LEARNING_RATE,
                                              seed=SEED, n_samples= NN_SAMPLES)

    #Method calculates Neurons position
    def get_NN_topology_size(self):
        for index in range(0, NEURON_LAYERS):
            list = []
            start_y = ((WINDOW_HEIGHT + NEURON_TOP_OFFSET) - (NEURONS[index] * NEURON_SIZE +
                                                              (NEURONS[index] - 1) * NEURON_Y_SPACE)) / 2

            start_x = NEURON_LEFT_POS + NEURON_X_SPACE * (index) + (index + 1) * NEURON_SIZE
            # print(f'Start X: {start_x} Start Y: {start_y}')

            for neuron in range(0, NEURONS[index]):
                list.append((start_x + NEURON_SIZE / 2, start_y + NEURON_SIZE / 2))  # coords for labels
                start_y += NEURON_Y_SPACE + NEURON_SIZE
            self.neurons_coords.append(list)

    #Method calculates synapses between Neurons
    def get_NN_synapses_size(self):
        lines = []

        for i in range(0, len(self.neurons_coords) - 1):
            base = self.neurons_coords[i]
            next = self.neurons_coords[i + 1]

            for j in range(0, len(base)):
                for k in range(0, len(next)):
                    # Add each line to list and to dictionary
                    lines.append([base[j][0] + NEURON_SIZE / 2 + NEURON_PEN_WIDTH, base[j][1],
                                  next[k][0] - NEURON_SIZE / 2 - NEURON_PEN_WIDTH, next[k][1]])
            # print(f'{i}: {lines}')
            self.weight_lines[i + 1] = lines
            lines = []

        # Output line
        out = self.neurons_coords[-1]
        self.weight_lines[NEURON_LAYERS] = [[out[0][0] + NEURON_SIZE / 2 + NEURON_PEN_WIDTH,
                                             out[0][1], out[0][0] + 150, out[0][1]]]

    #Method draw NN topology - layer architecture
    def draw_NN_topology(self, painter):

        pen_neuron = QPen(NEURON_PEN_COLOR, NEURON_PEN_WIDTH, NEURON_PEN_STYlE)
        painter.setPen(pen_neuron)

        for index in range(0,NEURON_LAYERS):
            #print(f'Start X: {start_x} Start Y: {start_y}')
            for neuron in range(0, NEURONS[index]):
                painter.drawEllipse(self.neurons_coords[index][neuron][0] - NEURON_SIZE/2,
                                    self.neurons_coords[index][neuron][1] - NEURON_SIZE/2,
                                    NEURON_SIZE,
                                    NEURON_SIZE)

    #Method draws NN layer synapses
    def draw_NN_synapses(self, painter, color=WEIGHT_PEN_COLOR):

        #Draw synapses
        pen_synapses = QPen(color, WEIGHT_PEN_WIDTH, WEIGHT_PEN_STYLE)
        painter.setPen(pen_synapses)

        for i in range(1, NEURON_LAYERS + 1):
            lines = self.weight_lines[i]

            for line in lines:
                #Draw line for each synapse between Neurons
                painter.drawLine(line[0], line[1],
                                 line[2], line[3])

    #Method draws NN architecture and caches it to Pixmap (Label)
    def draw_NN_architecture(self):
        painter = QPainter()
        painter.begin(self.pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        self.cache.clear()

        self.draw_NN_topology(painter)
        self.draw_NN_synapses(painter)

        painter.end()

        #Save to the cache
        self.cache.setPixmap(self.pixmap)

    def add_buttons(self):
        "Method add buttons to window"

        font = QFont("Calibri Light", 10)

        self.btn_init = QPushButton("Načítať NN", self)
        self.btn_init.move(1300, 550)
        self.btn_init.setStyleSheet("color: " + BUTTON_FONT_COLOR)
        self.btn_init.setFont(font)

        self.btn_train = QPushButton("Trénovať", self)
        self.btn_train.move(1300, 600)
        self.btn_train.setStyleSheet("color: " + BUTTON_FONT_COLOR)
        self.btn_train.setFont(font)

        self.btn_reset = QPushButton("Reset", self)
        self.btn_reset.move(1300, 650)
        self.btn_reset.setFont(font)
        self.btn_reset.setStyleSheet("color: " + BUTTON_FONT_COLOR)

        #Add slots to button clicked signal
        self.btn_init.clicked.connect(self.load_NN)
        self.btn_train.clicked.connect(self.train_NN)
        self.btn_reset.clicked.connect(self.reset_NN)


    #Event method for key shortcuts to buttons
    def keyPressEvent(self, event):

        #L = load NN
        if event.key() == Qt.Key_L:
            self.load_NN()
        #T = train NN
        elif event.key() == Qt.Key_T:
            self.train_NN()
        #R = reset NN
        elif event.key() == Qt.Key_R:
            self.reset_NN()

    #Method adds transparent background to the labels
    def add_label_background(self, label,
                             back_color=LBL_TRANSPARENT_BACKGROUND,
                             color=LBL_WEIGHT_FONT_COLOR,
                             text=None):
        if text:
            label.setText(text)
            label.adjustSize()
        label.setStyleSheet("background-color: " + back_color +
                            ";color: " + color)

    #Method creates and adds main title labels to the window
    def add_label(self, key, text, pos_x, pos_y, width, height,
                  font=TITLE_FONT, font_size=TITLE_FONT_SIZE,
                  color="black", bold=False, centered=False):
        self.titles[key] = QLabel(text, self)
        font = QFont(font, font_size)

        if bold:
            font.setBold(True)
        self.titles[key].setFont(font)
        self.titles[key].adjustSize()
        self.add_label_background(self.titles[key], color=color)

        if centered:
            self.titles[key].move(pos_x - self.titles[key].width()/2,
                                  pos_y - self.titles[key].height()/2)
        else:
            self.titles[key].move(pos_x, pos_y)

    #Method adds main titles to the window
    def add_titles(self):
        #Main title
        self.add_label(key='title', text=TITLE_MAIN, pos_x=WINDOW_WIDTH/2,
                       pos_y=30, width=300, height=30, font=TITLE_FONT,
                       font_size=20, color=TITLE_FONT_COLOR,
                       bold=False, centered=True)

        #Add layers titles
        for i in range(0, NEURON_LAYERS):
            if i == 0:
                left = NEURON_LEFT_POS + NEURON_SIZE
                self.add_label(key='input', text='Input', pos_x= left,
                               pos_y=200, width=150, height=30, font=TITLE_LAYERS_FONT,
                               font_size=TITLE_LAYERS_FONT_SIZE,
                               color=TITLE_LAYERS_FONT_COLOR, bold=False)
            elif i == NEURON_LAYERS-1:
                self.add_label(key='output', text='Output', pos_x=left,
                               pos_y=320, width=150, height=30, font=TITLE_LAYERS_FONT,
                               font_size=TITLE_LAYERS_FONT_SIZE,
                               color=TITLE_LAYERS_FONT_COLOR, bold=False)
            else:
                self.add_label(key=f'h_{i}', text=f'Hidden', pos_x=left,
                               pos_y=135, width=150, height=30, font=TITLE_LAYERS_FONT,
                               font_size=TITLE_LAYERS_FONT_SIZE,
                               color=TITLE_LAYERS_FONT_COLOR, bold=False)
            left += NEURON_X_SPACE + NEURON_SIZE

        self.add_label(key=f'result', text=f'Výstup', pos_x=1200,
                       pos_y=380, width=150, height=30, font=LBL_INFO_FONT,
                       font_size=LBL_INFO_FONT_SIZE,
                       color=LBL_INFO_FONT_COLOR, bold=False)
        self.add_label(key=f'expected', text=f'Očakávanie', pos_x=1300,
                       pos_y=380, width=150, height=30, font=LBL_INFO_FONT,
                       font_size=LBL_INFO_FONT_SIZE,
                       color=LBL_INFO_FONT_COLOR, bold=False)
        self.add_label(key=f'loss', text=f'MSE', pos_x=1450,
                       pos_y=380, width=150, height=30, font=LBL_INFO_FONT,
                       font_size=LBL_INFO_FONT_SIZE,
                       color=LBL_INFO_FONT_COLOR, bold=False)
        self.add_label(key=f'result_val', text='', pos_x=1200,
                       pos_y=420, width=150, height=30, font=TITLE_LAYERS_FONT,
                       font_size=12,
                       color=TITLE_LAYERS_FONT_COLOR, bold=False)
        self.add_label(key=f'expected_val', text='', pos_x=1300,
                       pos_y=420, width=150, height=30, font=TITLE_LAYERS_FONT,
                       font_size=12,
                       color=TITLE_LAYERS_FONT_COLOR,
                       bold=False)
        self.add_label(key=f'loss_val', text='', pos_x=1450,
                       pos_y=420, width=150, height=30, font=TITLE_LAYERS_FONT,
                       font_size=12,
                       color=TITLE_LAYERS_FONT_COLOR, bold=False)
        #Label status for training flow
        self.add_label(key=f'status', text='', pos_x=520,
                       pos_y=720, width=150, height=30, font=TITLE_LAYERS_FONT,
                       font_size=TITLE_LAYERS_FONT_SIZE,
                       color=LBL_INFO_FONT_COLOR,
                       bold=False, centered=True)

        # #Epoch label
        # self.add_label(key=f'epoch', text='Epocha', pos_x=1450,
        #                pos_y=440, width=150, height=30, font=LBL_INFO_FONT,
        #                font_size=LBL_INFO_FONT_SIZE,
        #                color=LBL_INFO_FONT_COLOR, bold=False)
        # self.add_label(key=f'epoch', text='', pos_x=1450,
        #                pos_y=500, width=150, height=30, font=LBL_INFO_FONT,
        #                font_size=LBL_INFO_FONT_SIZE,
        #                color=TITLE_LAYERS_FONT_COLOR, bold=False)

    #Method creates and adds labels for text in each neuron
    def add_neuron_labels(self):
        #Initialize empty dicts
        for i in range(0, NEURON_LAYERS):
            self.lbl_dots[i] = []

        #Add labels for each neuron
        for index, layer in enumerate(self.neurons_coords):
            for coords in layer:

                self.lbl_dots[index].append(QLabel(LBL_DOTS_PLACEHOLDER, self))
                if index == 0:
                    self.add_label_background(self.lbl_dots[index][-1], color=LBL_DOTS_INPUT_COLOR,
                                              text=LBL_DOTS_PLACEHOLDER)
                else:
                    self.add_label_background(self.lbl_dots[index][-1], color=LBL_INFO_FONT_COLOR,
                                              text=LBL_DOTS_PLACEHOLDER)

                font = QFont(LBL_DOTS_FONT, LBL_DOTS_FONT_SIZE)
                font.setBold(True)
                self.lbl_dots[index][-1].setFont(font)
                self.lbl_dots[index][-1].adjustSize()
                self.lbl_dots[index][-1].move(coords[0] - self.lbl_dots[index][-1].width()/2,
                                              coords[1] - self.lbl_dots[index][-1].height()/2)

        #print(self.weight_lines)

    #Method sets the weight label properties
    def set_weight_label_props(self, layer, index, x=None, y=None,
                               font=LBL_WEIGHT_FONT,
                               size=LBL_WEIGHT_FONT_SIZE,
                               color=LBL_WEIGHT_FONT_COLOR,
                               text=None):
        if x is not None and y is not None:
            self.weight_lbls[layer][index].move(x, y)

        #Set text to the label
        self.add_label_background(self.weight_lbls[layer][index],
                                  color=color,
                                  text=text)

    def add_weight_labels(self):
        #Initialize empty dicts for labels
        for i in range(1,NEURON_LAYERS):
            self.weight_lbls[i] = []

        #First three H1_1
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1,-1, 470, 180, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 425, 220, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 230, color=LBL_INFO_FONT_COLOR)

        #Second three H1_2
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1,-1, 470, 285, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 425, 310, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 330, color=LBL_INFO_FONT_COLOR)

        #Third three H1_3
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 385, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 425, 400, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 430, color=LBL_INFO_FONT_COLOR)

        # Fourth three H1_4
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 485, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 425, 490, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 530, color=LBL_INFO_FONT_COLOR)

        #Fifth three H1_5
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 595, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 425, 580, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[1].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(1, -1, 470, 640, color=LBL_INFO_FONT_COLOR)

        #Five of Output O1
        self.weight_lbls[2].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(2, -1, 760, 340, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[2].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(2, -1, 760, 370, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[2].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(2, -1, 760, 400, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[2].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(2, -1, 760, 425, color=LBL_INFO_FONT_COLOR)
        self.weight_lbls[2].append(QLabel(LBL_WEIGHT_PLACEHOLDER, self))
        self.set_weight_label_props(2, -1, 760, 450, color=LBL_INFO_FONT_COLOR)
        #Last one as result in dots!!

    #Method updates the Canvas to show the plot
    def update_plot(self):
        if self.reseted:
            return

        if not self.NN.mean_loss:
            return

        self.ydata = self.NN.mean_loss
        self.xdata = range(0,len(self.ydata))
        #Clear chart visual
        self.chart.ax.cla()

        #Plot the data
        self.chart.ax.plot(self.xdata, self.ydata)
        self.chart.set_chart_titles()

        #Redraw the chart
        self.chart.draw()

    #Method clears the Canvas
    def clear_plot(self):
        self.chart.ax.clear()
        self.chart.set_chart_titles()
        self.chart.draw()

    # Method draws synaptic weight line between Neurons with specific color
    def draw_custom_synapse(self, painter, layer, id, color, text=None):

        synapse_color = QColor()
        synapse_color.setNamedColor(color)  #Set HEX color format

        # Set painter pen for synapse
        pen_synapses = QPen(synapse_color, WEIGHT_PEN_WIDTH, WEIGHT_PEN_STYLE)
        painter.setPen(pen_synapses)
        painter.drawLine(*self.weight_lines[layer][id])

        if text:
            self.weight_lbls[layer][id].setText(text)

        self.add_label_background(self.weight_lbls[layer][id], color=LBL_INFO_FONT_COLOR)

    #Method loads inputs for NN
    def load_inputs(self):

        input_layer = 0
        item = self.NN.get_input()
        for input in range(0, NEURONS[0]):
            self.lbl_dots[input_layer][input].setText(f'{item[input]:.3f}')
            self.lbl_dots[input_layer][input].adjustSize()
            self.lbl_dots[input_layer][input].move(self.neurons_coords[0][input][0] -
                                                   self.lbl_dots[input_layer][input].width() / 2,
                                                   self.neurons_coords[0][input][1] -
                                                   self.lbl_dots[input_layer][input].height() / 2)

    #Method initializes NN
    def load_NN(self):

        #Firstly load input Labels!
        self.load_inputs()

        #Load weights and redraw topology
        t = self.reseted
        self.reseted = False
        self.load_NN_weights()
        self.reseted = t
        self.NN_loaded = True

    #Method loads weights from NN and sets them to the weight Labels
    def load_NN_weights(self, pallete=COLOR_PALLETE, trained=False):

        if self.reseted:
            return

        if trained:
            # Update info status label
            # Update info status label
            pixmap = QPixmap(STATUS_ARROWS[1])
            self.titles['status'].setPixmap(pixmap)
            self.titles['status'].resize(pixmap.width(), pixmap.height())
            self.titles['status'].move(550 - self.titles['status'].width(),
                                       720)

        self.pixmap = QPixmap(*PIXMAP_SIZE)
        self.pixmap.fill(Qt.white)
        self.cache.clear()

        painter = QPainter()
        painter.begin(self.pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_NN_topology(painter)
        self.draw_NN_synapses(painter)

        #Draw synaptic weight lines with colors
        for layer in range(1, NEURON_LAYERS):
            weights = len(self.weight_lbls[layer])
            nn_weights = self.NN.get_weights(layer).flatten()

            pallete = cmap.get_cmap(pallete, weights)

            for i in range(0, weights):
                color = pallete(nn_weights[i])
                color = colors.rgb2hex(color)
                self.draw_custom_synapse(painter, layer, i, color, f"{nn_weights[i]:.2f}")

        #Draw ouptut synapse
        pen_synapses = QPen(WEIGHT_OUTPUT_PEN_COLOR, WEIGHT_PEN_WIDTH, WEIGHT_PEN_STYLE)
        painter.setPen(pen_synapses)
        painter.drawLine(*self.weight_lines[3][0])

        painter.end()
        self.cache.setPixmap(self.pixmap)

    #Method resets NN architecture to default state
    def reset_NN(self):

        #Stop timer if is running
        if  self.timer:
            if self.timer.isActive():
                self.timer.stop()

        self.pixmap = QPixmap(*PIXMAP_SIZE)
        self.pixmap.fill(Qt.white)
        self.cache.clear()


        #Clear all dots labels
        for i in range(0, NEURON_LAYERS):
            for j in range(0, NEURONS[i]):
                self.lbl_dots[i][j].setText(LBL_DOTS_PLACEHOLDER)

        #Clear all weight labels
        for i in range(1, NEURON_LAYERS):
            lbls = self.weight_lbls[i]
            for lbl in lbls:
                lbl.setText(LBL_WEIGHT_PLACEHOLDER)

        self.clear_info_labels()

        #Redraw architecture
        self.draw_NN_architecture()
        #Reset flags and reload NN
        self.epoch_counter = 0
        self.NN_loaded = False
        self.reseted = True
        self.loss = None
        self.init_NN()
        self.clear_plot()

    #Method trains NN and highlights weights/labels/loss
    def train_NN(self):

        if not self.NN_loaded:
            return

        self.reseted = False
        self.timer = QTimer()
        self.timer.start(1000)

        self.load_inputs()
        self.NN.forward()
        clb = functools.partial(self.update_dots, layer=2)
        QTimer.singleShot(300, clb)
        clb = functools.partial(self.update_dots, layer=3)
        QTimer.singleShot(600, clb)

        clb = functools.partial(self.update_info_labels)
        QTimer.singleShot(900, clb)

        #Load NN new weights
        self.NN.backpropagation()
        clb = functools.partial(self.load_NN_weights)
        QTimer.singleShot(900, clb)
        self.NN_loaded = True

        #Draw Loss chart
        self.timer.timeout.connect(self.update_NN)

    #Method updates dots product for specific layer
    def update_dots(self, layer):
        if not self.NN_loaded:
            return

        #Update info status label
        pixmap = QPixmap(STATUS_ARROWS[0])
        self.titles['status'].setPixmap(pixmap)
        self.titles['status'].resize(pixmap.width(), pixmap.height())
        self.titles['status'].move(550 - self.titles['status'].width(),
                                   720)

        dots = self.lbl_dots[layer-1]
        NN_dots = self.NN.get_dots(layer).flatten()
        print(NN_dots)

        #Update lbl dots text and position
        for idx, dot in enumerate(dots):
            dot.setText(f'{NN_dots[idx]:.3f}')
            dot.adjustSize()
            dot.move(self.neurons_coords[layer-1][idx][0] - dot.width()/2,
                     self.neurons_coords[layer-1][idx][1] - dot.height()/2)

    #Method updates info labels
    def update_info_labels(self):
        if not self.NN_loaded:
            return

        loss = self.NN.loss[-1]
        expected = self.NN.O
        result = float(self.NN.Z[NEURON_LAYERS-1][-1])
        # print(f"Result is {result}, expe: {expected}, loss {loss}")

        self.titles['result_val'].setText(f'{result:.3f}')
        self.titles['result_val'].adjustSize()

        self.titles['expected_val'].setText(f'{expected:.3f}')
        self.titles['expected_val'].adjustSize()

        self.titles['loss_val'].setText(f'{loss:.3f}')
        self.titles['loss_val'].adjustSize()

    #Method clears info labels
    def clear_info_labels(self):
        self.titles['result_val'].setText(LBL_DOTS_PLACEHOLDER)
        self.titles['result_val'].adjustSize()

        self.titles['expected_val'].setText(LBL_DOTS_PLACEHOLDER)
        self.titles['expected_val'].adjustSize()

        self.titles['loss_val'].setText(LBL_DOTS_PLACEHOLDER)
        self.titles['loss_val'].adjustSize()

        self.titles['status'].clear()
        self.titles['status'].setText(LBL_DOTS_PLACEHOLDER)
        self.titles['status'].adjustSize()

    def update_NN(self):
        #Stop NN training after reaching epochs
        if self.epoch_counter >= EPOCHS-1:
            self.timer.stop()
            return

        self.NN.forward()
        self.load_inputs()

        clb = functools.partial(self.update_dots, layer=2)
        QTimer.singleShot(250, clb)
        clb = functools.partial(self.update_dots, layer=3)
        QTimer.singleShot(500, clb)

        # Draw Loss chart & update info labels
        clb = functools.partial(self.update_info_labels)
        QTimer.singleShot(750, clb)

        self.NN.backpropagation()
        clb = functools.partial(self.load_NN_weights, trained=True)
        QTimer.singleShot(750, clb)

        clb = functools.partial(self.update_plot)
        QTimer.singleShot(900, clb)

        self.epoch_counter += 1


#####################################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(APP_STYLE)
    m_controller = ViewController()
    sys.exit(app.exec_())