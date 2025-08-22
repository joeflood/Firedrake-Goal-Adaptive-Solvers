# -*- coding: utf-8 -*-
import sys
import os
import re
import glob

from setuptools import setup

version = re.findall('__version__ = "(.*)"',
                     open('defcon/__init__.py', 'r').read())[0]

packages = [
    "defcon",
    "defcon.cli",
    "defcon.gui",
    ]

CLASSIFIERS = """
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# TODO: This is cumbersome and prone to omit something
demofiles = glob.glob(os.path.join("examples", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.xml*"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.geo"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.xml*"))

# Don't bother user with test files
[demofiles.remove(f) for f in demofiles if "test_" in f]

if sys.version_info[0] == 2:
    entry_points = {'console_scripts': ['defcon = defcon.__main__:main',
                                        'defcon2 = defcon.__main__:main']}
else:
    entry_points = {'console_scripts': ['defcon = defcon.__main__:main',
                                        'defcon3 = defcon.__main__:main']}

setup(packages=packages,
      package_dir={"defcon": "defcon"},
      package_data={"defcon": ["gui/resources/*.png"]},
      data_files=[(os.path.join("share", "defcon", os.path.dirname(f)), [f])
                  for f in demofiles],
      entry_points=entry_points,
    )
