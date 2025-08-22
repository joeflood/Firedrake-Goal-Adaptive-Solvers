from __future__ import absolute_import, print_function

import os
import sys
import importlib.util

from defcon import BifurcationProblem
from defcon import ComplementarityProblem


def fetch_bifurcation_problem(path):
    """Try to load BifurifurcationProblem class and return its
    instance, either from given filename or directory, in which
    case filename is inferred from last directory component.

    When successful return BifurcationProblem isntance, otherwise
    None.
    """
    cwd = os.getcwd()
    probdir = os.path.dirname(path)
    if len(probdir) > 0:
        os.chdir(probdir)

    # Check if the user's given us a directory or a .py file
    if path.endswith(".py"):
        probpath = path
    elif os.path.isdir(path):
        path = os.path.abspath(path)
        if path.endswith(os.path.sep):
            path = path[:-1]

        lastname = path.split(os.path.sep)[-1]
        probpath = path + os.path.sep + lastname + ".py"
    else:
        print("Either specify the .py file or the directory containing it.")
        return None

    # Import file
    try:
        sys.path.append(os.getcwd())
        spec = importlib.util.spec_from_file_location("prob", probpath)
        prob = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prob)
    except Exception:
        print("Was not able to import '%s'" % probpath)
        print("Please provide correct problem path!")
        print("")
        print("Backtrace follows:")
        import traceback; traceback.print_exc()
        return None

    # Run through each class we've imported and figure out which one inherits from BifurcationProblem
    for v in vars(prob).values():
        try:
            # Check if we have what we want
            assert issubclass(v, BifurcationProblem)
            assert v is not BifurcationProblem
            assert v is not ComplementarityProblem
            print("Found BifurcationProblem subclass: %s" % v)

            # Now try to initialize it with no arguments
            try:
                problem = v()
            except Exception:
                print("Failed to init BifurcationProblem subclass: %s " % v)
                print("Backtrace follows:")
                import traceback; traceback.print_exc()
                print("Trying other classes...")
            else:
                break

        except (AssertionError, TypeError):
            # Did not pass the requirements, try other classes
            pass
    else:
        # Not good, the loop finished without break
        print("Failed to fetch bifurcation problem")
        return None

    problem._path = probpath

    return problem
