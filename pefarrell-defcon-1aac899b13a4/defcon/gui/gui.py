#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

import matplotlib

# Get the window and figure code.
# NOTE: The figure, colours and markers, and a couple of utility functions will be imported from here.
# If necessary, make changes in qtwindows.py.
from defcon.gui.qtwindows import *

from defcon.cli.common import fetch_bifurcation_problem
import defcon.backend as backend

import sys, getopt, os, inspect
import time as TimeModule

# Imports for the paraview and hdf5topvd methods.
from subprocess import Popen
from defcon.parametertools import parameters_to_string

# We'll use literal_eval to get lists and tuples back from the journal.
# This is not as bad as eval, as it only recognises: strings, bytes, numbers, tuples, lists, dicts, booleans, sets, and None.
# It can't do anything horrible if you pass it, for example, the string "os.system("rm -rf /")".
from ast import literal_eval

# For saving movies.
from matplotlib import animation

# For plotting solutions if we don't use paraview, as well as for creating mvoies.
import matplotlib.pyplot as plt

# Saving tikz pictures.
try:
    from tikzplotlib import save as tikz_save
    use_tikz = True
except Exception:
    issuewarning("Could not import the library tikzplotlib. You will unable to save the file as a .tikz.\nTo use this functionality, install tikzplotlib, eg with:\n     # pip install tikzplotlib")
    use_tikz = False


#####################
### Utility Class ###
#####################
class PlotConstructor():
    """ Class for handling everything to do with the bifurcation diagram plot. """

    def __init__(self, journal_path, solutions_dir, xscale, problem, io, plot_with_mpl, mesh):
        self.xscale = xscale
        self.problem = problem
        self.io = io
        self.plot_with_mpl = plot_with_mpl
        self.mesh = mesh

        self.points = [] # Keep track of the points we've found, so we can redraw everything if necessary. Also for annotation.
        self.pointers = [] # Pointers to each point on the plot, so we can remove them.

        self.maxtime = 0 # Keep track of the furthest we have got in time.
        self.time = 0 # Keep track of where we currently are in time.
        self.lines_read = 0 # Number of lines of the journal file we've read.

        self.paused = False # Are we updating we new points, or are we frozen in time?

        self.annotation_highlight = None # The point we've annotated.
        self.annotated_point = None # The (params, branchid) of the annotated point

        self.journal_path = journal_path
        self.solutions_dir = solutions_dir

        self.freeindex = None # The index of the free parameter.

        self.current_functional = 0 # The index of the functional we're currently on.

        self.teamstats = [] # The status of each team.

        self.sweep = 0 # The parameter value we've got up to.
        self.sweepline = None # A pointer to the line object that shows how far we've got up to.
        self.sweeplines = [] # Keep track of where the sweepline is at each time step, so we can draw it in the movie.

        self.changed = False # Has the diagram changed since our last update?

        self.start_time = 0 # The (system) time at which defcon started running.
        self.running = False # Is defcon running?

        self._have_stability = False

    def set_app_win(self, aw):
        """Set ApplicationWindow instance we get bound to"""
        self.aw = aw

    ## Private utility functions. ##
    def distance(self, x1, x2, y1, y2):
        """ Return the L2 distance between two points. """
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def setx(self, ax):
        """ Sets the xscale to the user defined variable. """
        try:
            if self.xscale is not None: ax.set_xscale(self.xscale)
        except Exception:
            issuewarning("User-provided xscale variable is not a valid option for matplotlib.")
            pass

    def redraw(self):
        """Clears the window, redraws the labels and the grid. """
        bfdiag.clear()

        # Redraw labels.
        bfdiag.set_xlabel(self.parameter_name)
        bfdiag.set_ylabel(self.functional_names[self.current_functional])
        bfdiag.grid(color=GRID)

        # Reset the x and y limits.
        ys = [point[1][self.current_functional] for point in self.points]
        try:
            bfdiag.set_xlim([self.minparam, self.maxparam]) # fix the limits of the x-axis
            bfdiag.set_ylim([min(ys), max(ys)]) # reset the y limits, to prevent stretching
        except Exception: pass

        # Redraw the sweepline.
        #self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)

        # Reset the scale of the x axis.
        self.setx(bfdiag)

    def animate(self, i):
        """ Utility function for animating a plot. """
        for j in range(0, self.dots_per_frame):
            try:
                xs, ys, branchid, teamno, cont = self.points_iter.next()
                x = float(xs[self.freeindex])
                y = float(ys[self.func_index])
                self.ax.plot(x, y, marker=CONTPLOT, color=MAIN, linestyle='None')
                #self.animsweep = self.sweeplines_iter.next()
                #'if self.animsweepline is not None: self.animsweepline.remove()
                #self.animsweepline = self.ax.axvline(x=self.animsweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)
            except StopIteration: return

        # Let's output a log of how we're doing, so the user can see that something is in fact being done.
        if i % 50 == 0: print("Completed %d/%d frames" % (i, self.frames))
        return

    ## Controls for moving backwards and forwards in the diagram, or otherwise manipulating it. ##
    def pause(self):
        """ Pause the drawing. """
        self.paused = True

    def unpause(self):
        """ Unpause the drawing. """
        self.paused = False

    def seen(self):
        """ Tell the PlotConstructor we've got all the new points. """
        self.changed = False

    def start(self):
        """ Go to Time=0 """
        if not self.paused: self.pause()
        self.time = 0
        if self.annotated_point is not None: self.unannotate()
        self.redraw()
        self.changed = True
        return self.time

    def end (self):
        """ Go to Time=maxtime. """
        if self.time < self.maxtime:
            for i in range(self.time, self.maxtime):
                xs, ys, branchid, teamno, cont = self.points[i]
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, CONTPLOT
                else: c, m = DEF, DEFPLOT
                self.pointers[i] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.time = self.maxtime
            self.changed = True
            if self.paused: self.unpause()
        return self.maxtime

    def back(self):
        """ Take a step backwards in time. """
        if not self.paused: self.pause()
        if self.time > 0:
            if self.annotated_point is not None: self.unannotate()
            self.time -= 1
            self.pointers[self.time][0].remove()
            self.changed = True
        return self.time

    def forward(self):
        """ Take a step forwards in time. """
        if not self.paused: self.pause()
        if self.time < self.maxtime:
            xs, ys, branchid, teamno, cont = self.points[self.time]
            x = float(xs[self.freeindex])
            y = float(ys[self.current_functional])
            if cont: c, m= MAIN, CONTPLOT
            else: c, m= DEF, DEFPLOT
            self.pointers[self.time] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')
            self.time += 1
            self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time

    def jump(self, t):
        """ Jump to time t. """
        if not self.paused: self.pause()
        if self.annotated_point is not None: self.unannotate()

        # Case 1: we're going backwards in time, and need to remove points.
        if t < self.time:
            for i in range(t, self.time):
                self.pointers[i][0].remove()

        # Case 2: we're going forwards in time, and need to re-plot some points.
        elif t > self.time:
            for i in range(self.time, t):
                xs, ys, branchid, teamno, cont = self.points[i]
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, CONTPLOT
                else: c, m= DEF, DEFPLOT
                self.pointers[i] = bfdiag.plot(x, y, marker=m, color=c, linestyle='None')

        self.time = t
        self.changed = True
        if self.time==self.maxtime: self.unpause()
        return self.time


    def switch_functional(self, funcindex):
        """ Change the functional being plotted. """
        self.current_functional = funcindex
        if self.annotated_point is not None: self.unannotate()
        self.redraw() # wipe the diagram.
        self.pointers = [] # create an array to handle the pointers

        # Redraw all points up to the current time.
        Xcont = []; Ycont = []
        Xdef = []; Ydef = []
        for j in range(0, self.time):
            xs, ys, branchid, teamno, cont = self.points[j]
            x = float(xs[self.freeindex])
            y = float(ys[self.current_functional])
            if cont:
                Xcont.append(x)
                Ycont.append(y)
            else:
                Xdef.append(x)
                Ydef.append(y)
        if Xcont:
            self.pointers.append(bfdiag.plot(Xcont, Ycont, marker=CONTPLOT, color=MAIN, linestyle='None'))
        if Xdef:
            self.pointers.append(bfdiag.plot(Xdef, Ydef, marker=DEFPLOT, color=DEF, linestyle='None'))
        self.changed = True

    ## Functions for getting new points and updating the diagram ##
    def grab_data(self):
        """ Get data from the file. """
        # If the file doesn't exist, just pass.
        try: pullData = open(self.journal_path, 'r').read()
        except Exception: pullData = None
        return pullData

    def update(self):
        """ Handles the redrawing of the graph. """

        # Update the run time.
        if self.running: self.aw.set_elapsed_time(TimeModule.time() - self.start_time) # if defcon is still going, update the timer

        # If we're in pause mode then do nothing more.
        if self.paused:
            return self.changed

        # If we're not paused, we draw all the points that have come in since we last drew something.
        else:
            # Get new points, if they exist. If not, just pass.
            pullData = self.grab_data()
            if pullData is not None:
                dataList = pullData.split('\n')

                # Is this is first time we've found data, get the information from the first line of the journal.
                if self.freeindex is None:
                    self.running = True

                    freeindex, self.parameter_name, functional_names, unicode_functional_names, nteams, teamsize, minparam, maxparam, othervalues, timestamp, is_vi = dataList[0].split(';')
                    self.othervalues = literal_eval(othervalues)
                    self.start_time = float(timestamp)
                    self.minparam = float(minparam)
                    self.maxparam = float(maxparam)
                    self.is_vi = literal_eval(is_vi)
                    self.teamsize = int(teamsize)

                    # Now that we know we're dealing with a VI or not, initialise I/O
                    self.V = self.problem.function_space(self.mesh)
                    self.io.setup(self.problem.parameters(), None, self.V)

                    # Set up info about what the teams are doing.
                    self.nteams = int(nteams)
                    for team in range(self.nteams): self.teamstats.append(('i', 'None', 'None'))
                    self.aw.set_teamstats(self.nteams)

                    # Info about functionals.
                    self.freeindex = int(freeindex)
                    self.functional_names = literal_eval(functional_names)
                    self.unicode_functional_names = literal_eval(unicode_functional_names)
                    self.aw.make_radio_buttons(self.unicode_functional_names)

                    # Set the labels and scales of the axes.
                    bfdiag.set_xlabel(self.parameter_name)
                    bfdiag.set_ylabel(self.functional_names[self.current_functional])
                    bfdiag.set_xlim([self.minparam, self.maxparam]) # fix the limits of the x-axis
                    bfdiag.autoscale(axis='y')
                    self.setx(bfdiag)

                dataList = dataList[1:] # exclude the first line from now on.

                # Plot new points after reading the file
                Xcont = []; Ycont = []
                Xdef = []; Ydef = []
                for eachLine in dataList[self.lines_read:]:
                    if len(eachLine) > 1:
                        if eachLine[0] == '$':
                            pass
                            # This line of the journal is telling us about the sweep line.
                            #self.sweep = float(eachLine[1:])
                            #if self.sweepline is not None: self.sweepline.remove()
                            #self.sweepline = bfdiag.axvline(x=self.sweep, linewidth=1, linestyle=SWEEPSTYLE, color=SWEEP)

                        elif eachLine[0] == '~':
                            # This line of the journal is telling us about what the teams are doing.
                            team, task, params, branchid, timestamp = eachLine[1:].split(';')
                            self.teamstats[int(team)] = (task, params, branchid)
                            self.aw.update_teamstats(self.teamstats)
                            # If this tells us the teams have quit, we know we're not getting any more new points.
                            if task == 'q':
                                self.running = False
                                self.aw.set_elapsed_time(float(timestamp) - self.start_time) # if defcon is still going, update the timer

                        else:
                            # This is a newly discovered point. Get all the information we need.
                            teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                            xs = literal_eval(newparams)
                            ys = literal_eval(functionals)
                            x = float(xs[self.freeindex])
                            y = float(ys[self.current_functional])

                            # Use different colours/plot styles for points found by continuation/deflation.
                            if literal_eval(cont):
                                Xcont.append(x)
                                Ycont.append(y)
                            else:
                                Xdef.append(x)
                                Ydef.append(y)

                            # Keep track of the points we've discovered, as well as the matplotlib objects.
                            self.points.append((xs, ys, int(branchid), int(teamno), literal_eval(cont)))
                            self.sweeplines.append(self.sweep)
                            self.time += 1
                        self.lines_read +=1

                # Plot new continuation points
                if Xcont:
                    self.pointers.append(bfdiag.plot(Xcont, Ycont, marker=CONTPLOT, color=MAIN, linestyle='None'))
                # Plot new deflation points
                if Xdef:
                    self.pointers.append(bfdiag.plot(Xdef, Ydef, marker=DEFPLOT, color=DEF, linestyle='None'))

                # Update the current time.
                self.changed = True
                self.maxtime = self.time
                self.aw.set_time(self.time)
                return self.changed

    ## Functions for handling annotation. ##
    def annotate(self, clickX, clickY):
        """ Annotate a point when clicking on it. """
        if clickX is None and clickY is None:
           if self.annotated_point is not None:
               self.unannotate()
           return True

        # Sets a clickbox.
        xs = [float(point[0][self.freeindex]) for point in self.points[:self.time]]
        ys = [float(point[1][self.current_functional]) for point in self.points[:self.time]]


        # Empty lists throw a fatal error (Python 3.4+ has a better solution)
        try:
            xlen = max(xs) - min(xs)
        except ValueError:
            xlen = 1

        try:
            ylen = max(ys) - min(ys)
        except ValueError:
            ylen = 1

        if xlen == 0: xlen = 1
        if ylen == 0: ylen = 1

        xtick = bfdiag.get_xticks()
        ytick = bfdiag.get_yticks()
        xtol = (xtick[1]-xtick[0])/(2)
        ytol = (ytick[1]-ytick[0])/(2)

        annotes = []

        # Find the point on the diagram closest to the point the user clicked.
        time = 1
        for xs, ys, branchid, teamno, cont in self.points[:self.time]:
             x = float(xs[self.freeindex])
             y = float(ys[self.current_functional])
             if ((clickX-xtol < x < clickX+xtol) and (clickY-ytol < y < clickY+ytol)):
                 annotes.append((self.distance(x/xlen, clickX/xlen, y/ylen, clickY/ylen), x, y, branchid, xs, teamno, cont, time)) # uses rescaled distance.
             time += 1

        if len(annotes) > 0:
            annotes.sort() # sorts by first element automatically
            distance, x, y, branchid, xs, teamno, cont, time = annotes[0]
            if self.annotated_point is not None:
                self.unannotate()

            # Plot the annotation, and keep a handle on all the stuff we plot so we can use/remove it later.
            self.annotation_highlight = bfdiag.scatter([x], [y], s=[50], marker='o', color=HIGHLIGHT) # Note: change 's' to make the highlight blob bigger/smaller

            print("Normalised distances to click point: %s" % [annote[0]/annotes[0][0] for annote in annotes[:10]])
            branchids = []
            for annote in annotes:
                if (annote[0]/annotes[0][0] < 1.05) and annote[1] == annotes[0][1]:
                    branchids.append(annote[3])

            self.annotated_point = (xs, branchids)

            # Try to fetch the stability
            try:
                stab = [d["stable"] for d in self.io.fetch_stability(xs,branchids)]
                self._have_stability = True
            except (RuntimeError, KeyError, IOError):
                stab = None
                self._have_stability = False

            if stab is not None:
                stabstr = "stability: %s\n" % stab
            else:
                stabstr = ""

            self.aw.set_output_box("branches:  %s\n%s\nevent: #%d\n\nx = %s\ny = %s" % (branchids, stabstr, time, x, y))
            self.changed = True
        return self.changed

    def unannotate(self):
        """ Remove annotation from the graph. """
        self.annotation_highlight.remove()
        self.annotation_highlight = None
        self.annotated_point = None
        self.aw.set_output_box("")
        self.changed = True
        return False

    def ploteigenvalues(self):
        if self.annotated_point is not None:
            (params, branchids) = self.annotated_point
            stabs = self.io.fetch_stability(params, branchids, fetch_eigenfunctions=False)

            for (stab, branchid) in zip(stabs, branchids):
                plt.figure()

                print("Eigenvalues for branch %s at parameters %s:" % (branchid, str(params)))
                for eval_ in stab["eigenvalues"]:
                    print("  %s" % eval_)

                evals = map(complex, stab["eigenvalues"])
                for eval_ in evals:
                    plt.plot(eval_.real, eval_.imag, 'bo')

                plt.title("Eigenvalues for branch %s at parameters %s" % (branchid, str(params)))
                plt.xlabel("Real component")
                plt.ylabel("Imaginary component")

            plt.show()

    def postprocess(self):
        """ Fetch a solution and call the user-specified postprocessing routine. """
        if self.annotated_point is not None:
            (params, branchids) = self.annotated_point
            ys = self.io.fetch_solutions(params, branchids)

            for (y, branchid) in zip(ys, branchids):
                self.problem.postprocess(y, params, branchid, self.aw)

    def plotbranch(self):
        """ Fetch all solutions associated with selected branchid. """

        if self.annotated_point is None:
            return

        (params, branchids) = self.annotated_point
        branchid = branchids[0]

        # Construct the dictionary of fixed parameters
        freeindex = self.freeindex
        others    = self.othervalues

        fixed = {}
        i = 0
        for (index, param) in enumerate(self.io.parameters):
            if index != freeindex:
                fixed[param[1]] = others[i]
                i += 1

        # Fetch parameters for which this branch is known
        paramss = self.io.known_parameters(fixed, branchid)

        # Create the file to which we will write these solutions.
        print("Rendering branchid %s to PVD ..." % branchid)
        pvd_filename = os.path.join(self.solutions_dir, "branchid=%s.pvd" % branchid)
        self.save_pvd(paramss, [branchid], pvd_filename)

        # Finally, launch paraview with the newly created file.
        # If this fails, issue a warning.
        try: self.problem.launch_paraview(pvd_filename)
        except Exception as e:
            issuewarning("Oops, something went wrong with launching paraview. Are you sure you have it installed and on your PATH? The error was:")
            print(str(e))

    def plotparameter(self):
        """ Fetch all solutions associated with selected parameter. """

        if self.annotated_point is None:
            return

        (params, _) = self.annotated_point

        known_branches = self.io.known_branches(params)

        # Create the file to which we will write these solutions.
        pvd_filename = os.path.join(self.solutions_dir, "parameters=%s.pvd" % (params,)).replace(", ", "_").replace(",", "_")
        print("Rendering parameters %s to PVD ..." % (params,))
        self.save_pvd([params], known_branches, pvd_filename)

        # Finally, launch paraview with the newly created file.
        # If this fails, issue a warning.
        try: self.problem.launch_paraview(pvd_filename)
        except Exception as e:
            issuewarning("Oops, something went wrong with launching paraview. Are you sure you have it installed and on your PATH? The error was:")
            print(str(e))

    def plot(self):
        """ Fetch a solution and plot it. If the solutions are 1D we use matplotlib, otherwise we use paraview. """
        if self.annotated_point is not None:
            # Get the solution from the IO module.
            (params, branchids) = self.annotated_point

            if self.plot_with_mpl:
                ys = self.io.fetch_solutions(params, branchids)
                try:
                    if backend.__name__ == "dolfin":
                        x = backend.interpolate(backend.Expression("x[0]", degree=1), self.V)
                    elif backend.__name__ == "firedrake":
                        x = backend.Function(self.V).interpolate(backend.SpatialCoordinate(self.mesh)[0])
                    # FIXME: For functions f other than CG1, we might need to sort both arrays so that x is increasing. Check this out!
                    for (y, branchid) in zip(ys, branchids):
                        plt.close() # Close existing plots
                        plt.plot(x.vector().get_local(), y.vector().get_local(), '-', linewidth=3, color='b')
                        plt.title("branch %s, params %s" % (branchid, params))
                        plt.axhline(0, color='k') # Plot a black line through the origin
                        plt.show(block=False) # False here means the window is non-blocking, so we may continue using the GUI while the plot shows.
                except RuntimeError as e:
                    issuewarning("Error plotting expression. Are your solutions numbers rather than functions? If so, this is why I failed. The error was:")
                    print(str(e))
                    pass
            else:
                # Make a directory to put solutions in, if it doesn't exist.
                try: os.mkdir(self.solutions_dir)
                except OSError: pass

                # Create the file to which we will write these solutions.
                pvd_filename = os.path.join(self.solutions_dir, "SOLUTION$params=%s$branchids=%s.pvd" \
                    % (parameters_to_string(self.io.parameters, params), str(branchids).replace(", ", "_"))).replace(",", "_")
                self.save_pvd([params], branchids, pvd_filename)

                # Finally, launch paraview with the newly created file.
                # If this fails, issue a warning.
                try: self.problem.launch_paraview(pvd_filename)
                except Exception as e:
                    issuewarning("Oops, something went wrong with launching paraview. Are you sure you have it installed and on your PATH? The error was:")
                    print(str(e))

    ## Functions for saving to disk ##
    def save_movie(self, filename, length, fps):
        """ Creates a matplotlib animation of the plotting up to the current maxtime. """

        # Fix the functional we're currently on, to avoid unpleasantness if we try and change it while the movie is writing.
        self.func_index = self.current_functional

        # Make an iterator of the points list and sweepline list.
        self.points_iter = iter(self.points)
        #self.sweeplines_iter = iter(self.sweeplines)

        #self.animsweepline = None

        # Set up the animated figure.
        self.anim_fig = plt.figure()
        self.ax = plt.axes()
        self.ax.clear()

        self.ax.set_xlabel(self.parameter_name)
        self.ax.set_xlim([self.minparam, self.maxparam]) # fix the x-limits
        self.setx(self.ax)

        self.ax.set_ylabel(self.functional_names[self.func_index])
        ys = [point[1][self.current_functional] for point in self.points]
        self.ax.set_ylim([min(ys), max(ys)]) # fix the y-limits

        # Work out how many frames we want.
        self.frames = length * fps

        # Sanity check: if the number of desired frames is greater than the number of points, better fix this by adjusting fps.
        if self.frames > self.maxtime:
            self.frames = self.maxtime
            fps = int(self.frames / float(length))

        self.dots_per_frame = int(ceil(float(self.maxtime) / self.frames))

        # Create and save the animation.
        print("Saving movie. This may take a while...")
        try:
            self.anim = animation.FuncAnimation(self.anim_fig, self.animate, frames=self.frames, interval=1)
            mywriter = animation.FFMpegWriter(fps=fps, bitrate=5000)
            self.anim.save(filename, fps=fps, dpi=200, bitrate=5000, writer=mywriter, extra_args=['-vcodec', 'libx264'])
            print("Movie saved.")
            return
        except Exception as e:
            issuewarning("Saving movie failed. Perhaps you don't have ffmpeg installed? Anyway, the error was:")
            print(str(e))
            return

    def save_tikz(self, filename):
        """ Save the bfdiag window as a tikz plot. """
        if use_tikz:
            fig = plt.figure()

            ax = plt.axes()
            ax.clear()

            # Set up the x-axis.
            ax.set_xlabel(self.parameter_name)
            ax.set_xlim([self.minparam, self.maxparam])
            self.setx(ax)

            # Set up the y-axis.
            ax.set_ylabel(self.functional_names[self.current_functional])
            ys = [point[1][self.current_functional] for point in self.points]
            ax.set_ylim([floor(min(ys)), ceil(max(ys))])

            for xs, ys, branchid, teamno, cont in self.points:
                x = float(xs[self.freeindex])
                y = float(ys[self.current_functional])
                if cont: c, m= MAIN, '.'
                else: c, m= DEF, 'o'
                ax.plot(x, y, marker=m, color=c, linestyle='None')
            tikz_save(filename)
            ax.clear()
        else: issuewarning("tikzplotlib not installed. I can't save to tikz!")

    def save_pvd(self, paramss, branchids, pvdname):
        if self.teamsize == 1 or backend.__name__ == "dolfin":
            pvd = backend.File(pvdname)
            for params in paramss:
                ys = self.io.fetch_solutions(params, branchids)
                for y in ys:
                    y.rename("Solution", "Solution")
                    self.problem.save_pvd(y, pvd, params)
        else:
            try:
                os.remove(pvdname)
            except OSError:
                pass

            args = ["mpiexec", "-n", str(self.teamsize), "defcon", "make-pvd", self.problem._path, self.io.directory, repr(paramss), repr(branchids), pvdname]
            print("Executing: ", " ".join(args))
            dump = Popen(args)
            dump.wait()

def main(argv):
    # Set some defaults.
    problem_path = None
    output_dir = "output"
    solutions_dir = "solutions"
    problem_path = '.'
    xscale = None
    plot_with_mpl = False # Whether we will try to plot solutions with matplotlib. If false, we use paraview.
    update_interval = 100 # Update interval for the diagram.
    resources_dir = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../resources")) # The directory with the icons, etc.
    resources_dir += os.path.sep # Assume trailing slash

    # Get commandline args.
    def usage():
        sys.exit("""Usage: %s -p <problem path> -o <defcon output directory> -i <update interval in ms> -x <x scale> [<working directory>]
    Arguments:
          The working directory. This is the location where your defcon problem script is. If not given, current working dir is used.
    Options:
          -p: The name of the script you use to run defcon. If not provided, this defaults to the last component of the working directory.
              i.e, if the working directory is 'defcon/examples/elastica', we assume the name of the problem scipt is 'elastica'.
          -o: The directory that defcon uses for its output. The defaults to the "output" subdir of the working dir.
          -s: The directory to save solutions in. When you use paraview to visualise a solution, this is where it is saved. Defaults to the "solutions" subdir of the output dir.
          -i: The update interval of the bifurcation diagram in milliseconds. Defaults to 100.
          -x: The scale of the x-axis of the bifurcation diagram. This should be a valid matplotlib scale setting, eg 'log'.
          -h: Display this message.
          """ % argv[0])

    try: myopts, args = getopt.getopt(argv[1:],"p:o:i:s:x:h")
    except Exception: usage()

    for o, a in myopts:
        if   o == '-p': problem_path = os.path.expanduser(a)
        elif o == '-o': output_dir = os.path.expanduser(a)
        elif o == '-s': solutions_dir = os.path.expanduser(a)
        elif o == '-i': update_interval = int(a)
        elif o == '-x': xscale = a
        elif o == '-h': usage()
        else          : usage()

    if len(args) not in [0, 1]:
        usage()

    # Current dir is default working dir
    if len(args) == 0:
        args.append(os.getcwd())

    # Get absolue path of the working dir from the last command line argument
    working_dir = os.path.realpath(os.path.expanduser(args[-1]))

    # Get absolute paths from (evetually relative) paths
    output_dir = os.path.join(working_dir, output_dir)
    solutions_dir = os.path.join(output_dir, solutions_dir)

    # Get absolute path of journal file
    journal_path = os.path.join(output_dir, "journal", "journal.txt")

    # Get directory or file path to problem file
    problem_path = os.path.join(working_dir, problem_path)

    # Debugging paths
    if False:
        print(working_dir)
        print(output_dir)
        print(problem_path)
        print(journal_path)
        print(solutions_dir)
        print(resources_dir)

    # Get the BifurcationProblem instance
    problem = fetch_bifurcation_problem(problem_path)
    if problem is None:
        usage()

    # Get the mesh.
    mesh = problem.mesh(backend.comm_world)

    # If the mesh is 1D, we don't want to use paraview.
    if backend.__name__ == "dolfin":
        if mesh.geometry().dim() < 2: plot_with_mpl = True
    else:
        if mesh.geometric_dimension() < 2: plot_with_mpl = True

    # Get the function space and set up the I/O module for fetching solutions.
    io = problem.io(prefix=output_dir)

    # Main loop
    qApp = QtGui.QApplication(args)
    pc = PlotConstructor(journal_path, solutions_dir, xscale, problem, io, plot_with_mpl, mesh)
    aw = ApplicationWindow(pc, update_interval, resources_dir, working_dir)
    pc.set_app_win(aw)
    aw.setWindowTitle("DEFCON")
    aw.setWindowIcon(QtGuiVersionWorkaround.QIcon(resources_dir + 'defcon_icon.png'))
    aw.show()
    return(qApp.exec_())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
