from __future__ import absolute_import
from __future__ import print_function

import matplotlib

# Argh. What a mess QT + Python is!
# There are three different Python libraries. Let's
# try them in order of priority.
qtmodules = ["PyQt5", "PyQt4", "PySide2", "PySide"]
mplbackends = dict(PyQt4="Qt4Agg", PyQt5="Qt5Agg", PySide="Qt4Agg")

for qtmodule in qtmodules:
    try:
        __import__(qtmodule)
        matplotlib.use(mplbackends[qtmodule])
        print("\nUsing QT module %s" % qtmodule)
        break
    except:
        import traceback; traceback.print_exc()
        pass
else:
    raise ImportError("Could not find any of %s" % qtmodules)

# QT compatibility for matplotlib.
from matplotlib.backends import qt_compat

# Decide which version of QT to use. What a mess!
old_qt_search = hasattr(qt_compat, "QT_API_PYSIDE")
if old_qt_search:
    if qt_compat.QT_API == qt_compat.QT_API_PYSIDE:
        from PySide import QtGui, QtCore
        from PySide import QtGui as QtGuiVersionWorkaround
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
    elif qt_compat.QT_API in [qt_compat.QT_API_PYQT, qt_compat.QT_API_PYQTv2]:
        from PyQt4 import QtGui, QtCore
        from PyQt4 import QtGui as QtGuiVersionWorkaround
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
    elif qt_compat.QT_API == qt_compat.QT_API_PYQT5:
        from PyQt5 import QtCore
        from PyQt5 import QtWidgets as QtGui # This could get really confusing for future development
        from PyQt5 import QtGui as QtGuiVersionWorkaround
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
    else:
        raise ImportError("Unknown matplotlib QT API type: %s" % qt_compat.QT_API)
else:
    if qt_compat.QT_API == qt_compat.QT_API_PYSIDE2:
        from PySide import QtGui, QtCore
        from PySide import QtGui as QtGuiVersionWorkaround
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT
    elif qt_compat.QT_API == qt_compat.QT_API_PYQT5:
        from PyQt5 import QtCore
        from PyQt5 import QtWidgets as QtGui # This could get really confusing for future development
        from PyQt5 import QtGui as QtGuiVersionWorkaround
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
    else:
        raise ImportError("Unknown matplotlib QT API type: %s" % qt_compat.QT_API)

# Styles for matplotlib.
# See matpoltlib.styles.available for options.
try:
    from matplotlib import style
    style.use('ggplot')
except AttributeError:
    import warnings
    warnings.warn("Update to the latest version of matplotlib to use styles.")
    pass

# For plotting the bifurcation diagram.
from matplotlib.figure import Figure

from datetime import timedelta
from numpy import sqrt, floor, ceil
import os

# Seaborn colour palette.
colour_palette = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
                  (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
                  (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
                  (0.5058823529411764, 0.4470588235294118, 0.6980392156862745),
                  (0.8, 0.7254901960784313, 0.4549019607843137),
                  (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

blue, green, red, purple, yellow, cyan = ['#%02x%02x%02x' % tuple([int(c*255) for c in col]) for col in colour_palette] # convert the RGB values to HEX values.

# Colours.
MAIN = 'black' # colour for regular points
DEF = blue # colour for points found via deflation
HIGHLIGHT = red # colour for selected points
GRID = 'white' # colour the for grid.
BORDER = 'black' # borders on the UI
SWEEP = red # colour for the sweep line

# Markers and various other styles.
CONTPLOT = '.' # marker to use for points found by continuation
DEFPLOT = 'o' # marker to use for points found by deflation
SWEEPSTYLE = 'dashed' # line style for the sweep line

# Set up the figure.
figure = Figure(figsize=(7,6), dpi=100)
bfdiag = figure.add_subplot(111)
bfdiag.grid(color=GRID)

#########################
### Utility Functions ###
#########################
def teamtext(job, params, branchid):
    """ Utility function for converting a team's job into a colour and a label. """
    if job == "d": colour, label = blue, 'Deflating from branch %s at params %s' % (branchid, params)
    if job == "c": colour, label = green, 'Continuing branch %s to params %s' % (branchid, params)
    if job == "s": colour, label = purple, 'Computing stability of branch %s at params %s' % (branchid, params)
    if job == "i": colour, label = yellow, 'Idle'
    if job == "q": colour, label = red, 'Quit'
    return colour, label

def issuewarning(text):
    """ Prints a red warning message to the console. """
    print("\033[91m[Warning] %s\033[00m\n"%text)

################################
### Custom matplotlib Figure ###
################################
class DynamicCanvas(FigureCanvas):
    """A canvas that updates itself with a new plot."""
    def __init__(self, pc, update_interval, parent=None, width=5, height=4, dpi=100):
        FigureCanvas.__init__(self, figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_figure)
        self.timer.start(update_interval)
        self.pc = pc

    def update_figure(self):
        self.timer.stop()
        redraw = self.pc.update() # grab new points.
        if redraw: self.draw() # if we've found something new, redraw the diagram.
        self.pc.seen() # tell the PlotConstructor that we've seen all the new points.
        self.timer.start()

class CustomToolbar(NavigationToolbar2QT):
    """ A custom matplotlib toolbar, so we can remove those pesky extra buttons. """
    def __init__(self, canvas, parent, pc, resources_dir, working_dir):
        # Define which buttons we want and then initialise the toolbar.
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            ('Save', 'Save the figure', 'filesave', 'save_figure'),
            )
        NavigationToolbar2QT.__init__(self, canvas, parent)
        self.layout().takeAt(4) # This removes an annoying extra button at the end.

        self.setParent(parent)
        self.pc = pc
        self.working_dir = working_dir

        # Add new buttons for saving movies and saving to tikz.
        self.buttonSaveMovie = self.addAction(QtGuiVersionWorkaround.QIcon(resources_dir + "save_movie.png"), "Save Movie", self.save_movie)
        self.buttonSaveMovie.setToolTip("Save the figure as an animation")

        self.buttonSaveTikz= self.addAction(QtGuiVersionWorkaround.QIcon(resources_dir + "save_tikz.png"), "Save Tikz", self.save_tikz)
        self.buttonSaveTikz.setToolTip("Save the figure as tikz")

    def save_movie(self):
        """ A method that saves an animation of the bifurcation diagram. """
        start = self.working_dir + os.path.sep + "bfdiag.mp4" # default name of the file.
        filters = "FFMPEG Video (*.mp4)" # what kinds of file extension we allow.
        selectedFilter = filters

        # Ask for some input parameters.
        inputter = MovieDialog(self.parent)
        inputter.exec_()
        length = inputter.length.text()
        fps = inputter.fps.text()
        if inputter.create_movie:
            fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
            if fname:
                # normalise
                if isinstance(fname, tuple):
                    fname = fname[0]
                fname = str(fname)

                try:
                    self.pc.save_movie(fname, int(length), int(fps))
                # Handle any exceptions by printing a dialogue box
                except Exception as e:
                    QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)

    def save_tikz(self):
        """ A method that saves a tikz image of the bifurcation diagram. """
        start = self.working_dir + os.path.sep + "bfdiag.tex"
        filters = "Tikz Image (*.tex)"
        selectedFilter = filters
        fname = QtGui.QFileDialog.getSaveFileName(self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            # normalise
            if isinstance(fname, tuple):
                fname = fname[0]
            fname = str(fname)

            try:
                self.pc.save_tikz(fname)
            except Exception as e:
                QtGui.QMessageBox.critical(self, "Error saving file", str(e), QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)


############################
### MOVIE INPUT DIALOGUE ###
############################
class MovieDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MovieDialog, self).__init__(parent)

        # Layout
        mainLayout = QtGui.QVBoxLayout()

        lengthLayout = QtGui.QHBoxLayout()
        self.label = QtGui.QLabel()
        self.label.setText("Desired length of movie in seconds")
        lengthLayout.addWidget(self.label)

        self.length = QtGui.QLineEdit("60")
        self.length.setFixedWidth(80)
        lengthLayout.addWidget(self.length)
        mainLayout.addLayout(lengthLayout)

        fpsLayout = QtGui.QHBoxLayout()
        self.label2 = QtGui.QLabel()
        self.label2.setText("Frames per second")
        fpsLayout.addWidget(self.label2)

        self.fps = QtGui.QLineEdit("24")
        self.fps.setFixedWidth(80)
        fpsLayout.addWidget(self.fps)
        mainLayout.addLayout(fpsLayout)

        # Enter Button
        self.create_movie = False
        layout = QtGui.QHBoxLayout()
        button = QtGui.QPushButton("Enter")
        button.setFixedWidth(80)
        button.clicked.connect(self.save_movie)
        layout.addWidget(button)

        mainLayout.addLayout(layout)
        self.setLayout(mainLayout)

        self.resize(400, 60)
        self.setWindowTitle("Movie parameters")

    def save_movie(self):
        self.create_movie = True
        self.close()


######################
### Main QT Window ###
######################
class ApplicationWindow(QtGui.QMainWindow):
    def __init__(self, pc, update_interval, resources_dir, working_dir):
        QtGui.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pc = pc

        # Use this to add a toolbar, if desired.
        #self.file_menu = QtGui.QMenu('&File', self)
        #self.file_menu.addAction('&Quit', self.fileQuit, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        #self.menuBar().addMenu(self.file_menu)
        #self.help_menu = QtGui.QMenu('&Help', self)
        #self.menuBar().addSeparator()
        #self.menuBar().addMenu(self.help_menu)
        #self.help_menu.addAction('&About', self.about)

        # Main widget
        self.main_widget = QtGui.QWidget(self)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        # Keep track of the current time and maxtime.
        self.time = 0
        self.maxtime = 0

        # Layout
        main_layout = QtGui.QHBoxLayout(self.main_widget)
        lVBox = QtGui.QVBoxLayout()
        rVBox = QtGui.QVBoxLayout()
        rVBox.setAlignment(QtCore.Qt.AlignTop)
        main_layout.addLayout(lVBox)
        main_layout.addLayout(rVBox)

        canvasBox = QtGui.QVBoxLayout()
        lVBox.addLayout(canvasBox)
        timeBox = QtGui.QHBoxLayout()
        timeBox.setAlignment(QtCore.Qt.AlignCenter)
        lVBox.addLayout(timeBox)
        lowerBox = QtGui.QHBoxLayout()
        lowerBox.setAlignment(QtCore.Qt.AlignLeft)
        lVBox.addLayout(lowerBox)

        self.functionalBox = QtGui.QVBoxLayout()
        rVBox.addLayout(self.functionalBox)
        infoBox = QtGui.QVBoxLayout()
        infoBox.setContentsMargins(0, 10, 0, 10)
        rVBox.addLayout(infoBox)
        plotBox = QtGui.QVBoxLayout()
        rVBox.addLayout(plotBox)
        plotBox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        self.teamBox = QtGui.QVBoxLayout()
        self.teamBox.setContentsMargins(0, 10, 0, 10)
        rVBox.addLayout(self.teamBox)

        # Canvas.
        self.dc = DynamicCanvas(self.pc, update_interval, parent=self.main_widget, width=5, height=4, dpi=100)
        canvasBox.addWidget(self.dc)
        self.dc.mpl_connect('button_press_event', self.clicked_diagram)

        # Create the custom toolbar.
        toolbar = CustomToolbar(self.dc, self, self.pc, resources_dir, working_dir)
        toolbar.update()
        canvasBox.addWidget(toolbar)

        # Time navigation buttons
        self.buttonStart = QtGui.QPushButton()
        self.buttonStart.setIcon(QtGuiVersionWorkaround.QIcon(resources_dir+'start.png'))
        self.buttonStart.setIconSize(QtCore.QSize(18,18))
        self.buttonStart.clicked.connect(lambda:self.start())
        self.buttonStart.setFixedWidth(30)
        self.buttonStart.setToolTip("Start")
        timeBox.addWidget(self.buttonStart)

        self.buttonBack = QtGui.QPushButton()
        self.buttonBack.setIcon(QtGuiVersionWorkaround.QIcon(resources_dir+'back.png'))
        self.buttonBack.setIconSize(QtCore.QSize(18,18))
        self.buttonBack.clicked.connect(lambda:self.back())
        self.buttonBack.setFixedWidth(30)
        self.buttonBack.setToolTip("Back")
        timeBox.addWidget(self.buttonBack)

        self.jumpInput = QtGui.QLineEdit()
        self.jumpInput.setText(str(self.time))
        self.jumpInput.setFixedWidth(40)
        self.inputValidator = QtGuiVersionWorkaround.QIntValidator(self)
        self.inputValidator.setRange(0, self.maxtime)
        self.jumpInput.setValidator(self.inputValidator)
        self.jumpInput.returnPressed.connect(self.jump)
        timeBox.addWidget(self.jumpInput)

        self.buttonForward = QtGui.QPushButton()
        self.buttonForward.setIcon(QtGuiVersionWorkaround.QIcon(resources_dir+'forward.png'))
        self.buttonForward.setIconSize(QtCore.QSize(18,18))
        self.buttonForward.clicked.connect(lambda:self.forward())
        self.buttonForward.setToolTip("Forward")
        self.buttonForward.setFixedWidth(30)
        timeBox.addWidget(self.buttonForward)

        self.buttonEnd = QtGui.QPushButton()
        self.buttonEnd.setIcon(QtGuiVersionWorkaround.QIcon(resources_dir+'end.png'))
        self.buttonEnd.setIconSize(QtCore.QSize(18,18))
        self.buttonEnd.clicked.connect(lambda:self.end())
        self.buttonEnd.setToolTip("End")
        self.buttonEnd.setFixedWidth(30)
        timeBox.addWidget(self.buttonEnd)

        # Plot Buttons
        self.buttonPlot = QtGui.QPushButton("Plot selection")
        self.buttonPlot.clicked.connect(self.plot)
        self.buttonPlot.setEnabled(False)
        self.buttonPlot.setToolTip("Plot currently selected solutions")
        self.buttonPlot.setFixedWidth(120)
        plotBox.addWidget(self.buttonPlot)

        self.buttonPlotBranch = QtGui.QPushButton("Plot branch")
        self.buttonPlotBranch.clicked.connect(self.plotbranch)
        self.buttonPlotBranch.setEnabled(False)
        self.buttonPlotBranch.setToolTip("Plot selected branch")
        self.buttonPlotBranch.setFixedWidth(120)
        plotBox.addWidget(self.buttonPlotBranch)

        self.buttonPlotParameter = QtGui.QPushButton("Plot parameter")
        self.buttonPlotParameter.clicked.connect(self.plotparameter)
        self.buttonPlotParameter.setEnabled(False)
        self.buttonPlotParameter.setToolTip("Plot all solutions for selected parameter")
        self.buttonPlotParameter.setFixedWidth(120)
        plotBox.addWidget(self.buttonPlotParameter)

        self.buttonPostProcess = QtGui.QPushButton("Postprocess")
        self.buttonPostProcess.clicked.connect(self.postprocess)
        self.buttonPostProcess.setEnabled(False)
        self.buttonPostProcess.setToolTip("Execute user-specified postprocessing")
        self.buttonPostProcess.setFixedWidth(120)
        plotBox.addWidget(self.buttonPostProcess)

        self.buttonPlotEigenvalues = QtGui.QPushButton("Plot eigenvalues")
        self.buttonPlotEigenvalues.clicked.connect(self.ploteigenvalues)
        self.buttonPlotEigenvalues.setEnabled(False)
        self.buttonPlotEigenvalues.setToolTip("Plot eigenvalues")
        self.buttonPlotEigenvalues.setFixedWidth(120)
        plotBox.addWidget(self.buttonPlotEigenvalues)

        # Radio buttons
        label = QtGui.QLabel("Functionals:")
        label.setFixedHeight(20)
        self.functionalBox.addWidget(label)
        self.radio_buttons = []

        # Output Box
        self.infobox = QtGui.QLabel("")
        self.infobox.setFixedHeight(250)
        self.infobox.setFixedWidth(350)
        self.infobox.setAlignment(QtCore.Qt.AlignTop)
        font = QtGuiVersionWorkaround.QFont("Monospace")
        font.setStyleHint(font.TypeWriter)
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(75)
        self.infobox.setFont(font)
        self.infobox.setStyleSheet('border-color: %s; border-style: outset; border-width: 2px' % BORDER)
        infoBox.addWidget(self.infobox)

        # Teamstats Box
        label = QtGui.QLabel("Team Status:")
        label.setFixedHeight(20)
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.teamBox.addWidget(label)

        self.teamLabels = [] # A list of labels, one for each team.

        # Elapsed time counter.
        self.elapsedTime = QtGui.QLabel("Runtime: 0:00:00")
        self.elapsedTime.setAlignment(QtCore.Qt.AlignLeft)
        lowerBox.addWidget(self.elapsedTime)


    def set_time(self, t):
        """ Set the time, and also update the limits of time jump box if we need to. """
        if not t == self.time:
            self.time = t
            self.jumpInput.setText(str(self.time))
        # If this is larger than the current maxtime, update both the variable and the validator
        if t > self.maxtime:
            self.maxtime = t
            self.inputValidator.setRange(0, self.maxtime)

    def set_output_box(self, text):
        """ Set the text describing our annotated point. """
        self.infobox.setText(text)

    def set_teamstats(self, nteams):
        """ Do the intial setup for the grid of team status boxes. """
        # We'll define a square grid large enough to hold a box for each team.
        boxes = int(ceil(sqrt(nteams)))
        teamGrid = QtGui.QGridLayout()

        # Create a box for each team, going first along each column and then down a
        # row when we fill the column up.
        teamno, col, row = 0, 0, 0
        while teamno < nteams:
            while col <= boxes-1 and teamno < nteams:
                # Create a label,
                label = QtGui.QLabel(str(teamno))
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet("QLabel{font-size:40px;background-color:%s;border: 2px solid %s}" % (teamtext('i', 'None', 'None')[0], BORDER))
                size = int(floor(250.0/(1.5*boxes)))
                label.setFixedSize(size, size)

                # Add this label to a list so we can access it later, then put it into the
                # appropriate cell on the grid.
                self.teamLabels.append(label)
                teamGrid.addWidget(label, row, col)
                teamno += 1
                col += 1
            # We got to the end of the row. Go down a row and go back to the first column.
            row += 1
            col = 0
        teamGrid.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter)
        self.teamBox.addLayout(teamGrid)


    def update_teamstats(self, teamstats):
        """ Update the boxes that tell us what each team is doing. """
        for i in range(len(teamstats)):
            job, params, branchid = teamstats[i]
            colour, label = teamtext(job, params, branchid)
            self.teamLabels[i].setStyleSheet("QLabel {font-size:30px;background-color:%s;border: 2px solid %s}" % (colour, BORDER))
            self.teamLabels[i].setToolTip(("Team %s: " % i) + label)

    def make_radio_buttons(self, functionals):
        """ Build the radiobuttons for switching functionals. """
        for i in range(len(functionals)):
            # For each functional, make a radio button and link it to the switch_functionals method.
            radio_button = QtGui.QRadioButton(text=functionals[i])
            radio_button.clicked.connect(lambda: self.switch_functional())
            self.functionalBox.addWidget(radio_button)
            self.radio_buttons.append(radio_button)
        self.radio_buttons[0].setChecked(True) # Select the radio button corresponding to functional 0.

    def switch_functional(self):
        """ Switch functionals. Which one we switch to depends on the radiobutton clicked. """
        i = 0 # Keep track of the index of the radiobutton we're considering.
        for rb in self.radio_buttons:
            if rb.isChecked():
                # If this is the radio button that has been clicked, switch to the appropriate functional and jump out of the loop.
                self.pc.switch_functional(i)
                break
            else: i+=1

    def clicked_diagram(self, event):
        """ Annotates the diagram, by plotting a tooltip with the params and branchid of the point the user clicked.
            If the diagram is already annotated, remove the annotation. """
        annotated = self.pc.annotate(event.xdata, event.ydata)
        if annotated:
            self.buttonPlot.setEnabled(True)
            self.buttonPlotBranch.setEnabled(True)
            self.buttonPlotParameter.setEnabled(True)
            self.buttonPostProcess.setEnabled(True)
            if self.pc._have_stability:
                self.buttonPlotEigenvalues.setEnabled(True)
            else:
                self.buttonPlotEigenvalues.setEnabled(False)
        else:
            self.buttonPlot.setEnabled(False)
            self.buttonPlotBranch.setEnabled(False)
            self.buttonPlotParameter.setEnabled(False)
            self.buttonPostProcess.setEnabled(False)
            self.buttonPlotEigenvalues.setEnabled(False)

    def start(self):
        """ Set Time=0. """
        t = self.pc.start()
        self.set_time(t)

    def back(self):
        """ Set Time=Time-1. """
        t = self.pc.back()
        self.set_time(t)

    def forward(self):
        """ Set Time=Time+1. """
        t = self.pc.forward()
        self.set_time(t)

    def end(self):
        """ Set Time=Maxtime. """
        t = self.pc.end()
        self.set_time(t)

    def jump(self):
        """ Jump to Time=t. """
        t = int(self.jumpInput.text())
        new_time = self.pc.jump(t)
        self.set_time(new_time)

    def plot(self):
        """ Launch Matplotlib/Paraview to graph the selected solution. """
        self.pc.plot()

    def plotbranch(self):
        """ Launch Matplotlib/Paraview to graph the selected branch. """
        self.pc.plotbranch()

    def plotparameter(self):
        """ Launch Matplotlib/Paraview to graph all branches for the selected parameter. """
        self.pc.plotparameter()

    def postprocess(self):
        """ Do whatever postprocessing the user asks. """
        self.pc.postprocess()

    def ploteigenvalues(self):
        self.pc.ploteigenvalues()

    def set_elapsed_time(self, elapsed):
        """ Gets the amount of time that has elapsed since defcon started running. """
        t = str(timedelta(seconds=elapsed)).split('.')[0]
        self.elapsedTime.setText("Runtime: " + t)

