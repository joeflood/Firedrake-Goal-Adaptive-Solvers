from __future__ import absolute_import

from defcon.graph import DefconGraph
from defcon.tasks import DeflationTask

def test_waiting():
    graph = DefconGraph()

    task1 = DeflationTask(taskid=1,
                          oldparams=(0, 0),
                          freeindex=0,
                          branchid=0,
                          newparams=(1, 0))
    graph.wait(1, 0, task1)
    task2 = DeflationTask(taskid=2,
                          oldparams=(0, 0),
                          freeindex=0,
                          branchid=1,
                          newparams=(1, 0))
    graph.wait(2, 1, task2)

    assert len(graph.waiting(DeflationTask)) == 2
