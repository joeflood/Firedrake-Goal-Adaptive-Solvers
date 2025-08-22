#!/usr/bin/env python
from __future__ import absolute_import, print_function

import sys


def usage():
    print("""
Usage: defcon <command> [options] [args]

Commands:
  gui
  make-pvd
  subset-output
  merge-outputs
  recompute-functionals
  recompute-known-branches
  recompute-stability
  postprocess
  compute-distances
  plot-stability

Help:
  defcon <command> -h
""")


def main(args=None):
    """This is the commandline tools of the defcon package.
    If you want to call this routine directly (e.g. for purpose of
    testing) use sys.argv-like args:
    `(<script-name>, <command>, <arg1>, <arg2>, ...)`,
    e.g.::

        main(args=("defcon", "merge-outputs", "/path/to/my/problem.py",
                   "/path/to/output1", "/path/to/output2"))

    First argument `<script-name>` is actually ignored.
    """
    if args is None:
        args = sys.argv

    if len(args) <= 1:
        print("No command specified.")
        usage()
        return 1

    # Extract command and strip script name from args
    cmd = args[1]
    args  = ["defcon "+args[1]] + args[2:]

    # NOTE: We only import defcon.gui or defcon.cli if needed in the elif
    #       branches below before exiting; the code probably has side effects
    if cmd == "gui":
        import defcon.gui
        return defcon.gui.main(args)
    elif cmd == "make-pvd":
        import defcon.cli.makepvd
        return defcon.cli.makepvd.main(args)
    elif cmd == "subset-output":
        import defcon.cli.subsetoutput
        return defcon.cli.subsetoutput.main(args)
    elif cmd == "merge-outputs":
        import defcon.cli.mergeoutputs
        return defcon.cli.mergeoutputs.main(args)
    elif cmd == "recompute-functionals":
        import defcon.cli.recomputefunctionals
        return defcon.cli.recomputefunctionals.main(args)
    elif cmd == "recompute-known-branches":
        import defcon.cli.recomputeknownbranches
        return defcon.cli.recomputeknownbranches.main(args)
    elif cmd == "recompute-stability":
        import defcon.cli.recomputestability
        return defcon.cli.recomputestability.main(args)
    elif cmd == "postprocess":
        import defcon.cli.postprocess
        return defcon.cli.postprocess.main(args)
    elif cmd == "compute-distances":
        import defcon.cli.computedistances
        return defcon.cli.computedistances.main(args)
    elif cmd == "plot-stability":
        import defcon.cli.plotstability
        return defcon.cli.plotstability.main(args)
    else:
        print("Unknown command specified.")
        usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
