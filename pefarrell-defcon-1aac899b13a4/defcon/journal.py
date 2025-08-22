from __future__ import absolute_import

import time
import os
from ast import literal_eval

from defcon.tasks import ArclengthTask, StabilityTask, ContinuationTask, DeflationTask


codes = {ArclengthTask: "a", StabilityTask: "s", ContinuationTask: "c", DeflationTask: "d"}
def task_to_code(task):
    return codes[task.__class__]

class Journal(object):
    """
    Base class for Journal implementations.
    """
    def setup(self, nteams, teamsize, minparam, maxparam):
        raise NotImplementedError

    def entry(self, oldparams, branchid, newparams, functionals, continuation):
        raise NotImplementedError

    def sweep(self, params):
        raise NotImplementedError

    def team_job(self, team, task, params=None, branch=None):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError


class FileJournal(Journal):
    """
    Class that implements journalling using a single file, where the first line tells us some one-off
    details about a problem, and the subsequent lines each contain one piece of information.
    """
    def __init__(self, directory, parameters, functionals, freeindex, sign, is_vi):
        self.directory = directory + os.path.sep + "journal"
        self.parameters = parameters
        self.functionals = functionals
        self.freeindex = freeindex
        self.sign = sign
        self.sweep_params = None
        self.is_vi = is_vi

    def setup(self, nteams, teamsize, minparam, maxparam):
        """ Create the journal file and write the initial line of information. """
        # Create the journal file and directory
        try: os.mkdir(self.directory)
        except OSError: pass

        # Write all the necessary information about the problem, the axis labels and extent, the number of teams, etc.
        xlabel = self.parameters[self.freeindex][2]
        ylabels = [func[2] for func in self.functionals]
        unicodeylabels = [func[1] for func in self.functionals]

        # Other parameter values
        others = tuple(float(val[0]) for (i, val) in enumerate(self.parameters) if i != self.freeindex)

        try:
            os.makedirs(self.directory)
        except OSError:
            pass

        with open(self.directory + os.path.sep + "journal.txt", 'w') as f:
            f.write("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s\n" % (self.freeindex, xlabel, ylabels, unicodeylabels, nteams, teamsize, minparam, maxparam, others, time.time(), self.is_vi))

    def entry(self, teamid, oldparams, branchid, newparams, functionals, continuation):
        """ Tell the journal about a new point we've discovered. """
        # Workaround for numpy 2.0
        if oldparams is not None:
            oldparams = [float(y) for y in oldparams]
        if newparams is not None:
            newparams = [float(y) for y in newparams]
        with open(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("%s;%s;%s;%s;%s;%s \n" % (teamid, oldparams, branchid, newparams, functionals, continuation))

    def sweep(self, params):
        """ Tell the journal about an update to the sweepline. """
        self.sweep_params = params
        with open(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("$%.20f\n" % params)

    def team_job(self, team, task, params=None, branch=None):
        """ Tell the journal about what this team is doing. """
        # Workaround for numpy 2.0
        if params is not None:
            params = [float(y) for y in params]
        with open(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("~%s;%s;%s;%s;%s\n" % (team, task, params, branch, time.time()))

    def exists(self):
        """ Check if the journal file exists. """
        return os.path.exists(self.directory)

    def resume(self):
        """ Read the journal file and find the furthest extend of the branches we've discovered so far, as well as the position of the sweepline."""
        # Read data from the file.
        pullData = open(self.directory + os.path.sep + "journal.txt", 'r').read().split('\n')

        branches = dict()
        sweep = None

        freeindex = pullData[0].split(';')[0]
        others = literal_eval(pullData[0].split(';')[-2])

        for eachLine in pullData[1:]:
            if len(eachLine) > 1:
                if eachLine[0] == '$':
                    # This line tells us how far the sweep has gone.
                    sweep = float(eachLine[1:])
                elif eachLine[0] == '~':
                    # This line tells us something about what task one of the teams was doing, which is unimportant.
                    pass
                else:
                    # This tells us about a point we discovered.
                    teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                    branchid = int(branchid)
                    params = literal_eval(newparams)
                    branches[branchid] = (tuple([float(param) for param in params]), cont)

        # Remove any branches that consist of only one point found by deflation,
        # which avoids a bug whereby a point might be discovered by deflation
        # and written to the journal, but not saved by the IO module.
        for branchid in branches.keys():
            if not branches[branchid][1]: del branches[branchid]

        # Strip the 'cont' values out of the branches dictionary.
        branches = dict([(key, branches[key][0]) for key in branches.keys()])

        self.sweep_params = sweep
        assert(branches) # assert that branches is a nonempty dictionary.
        return (sweep, branches, int(freeindex), others)
