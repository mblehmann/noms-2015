#!/usr/bin/python

import random
from igraph import *

done = False
for i in range(100):
 g = Graph.GRG(33, 0.20)
 #g = Graph.Tree(81, 5)
 if g.is_connected():
  done = True
  break

if not done:
 print 'Failed'
 sys.exit(1)

for node in g.vs():
 node['type'] = "Router"

plot(g, sys.argv[1] + ".pdf")

print g.summary()

g.write_gml(sys.argv[1] + ".gml")

print 'Success'
