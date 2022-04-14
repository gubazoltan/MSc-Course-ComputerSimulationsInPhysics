#!/usr/bin/python
import numpy as np

def printsvg(x,y,gene,N,outfile):
  f = open(outfile,"w");
  f.write("<?xml version=\"1.0\" standalone=\"no\"?>\n")
  f.write("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n")
  f.write("\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n")
  f.write("<svg    xmlns=\"http://www.w3.org/2000/svg\"\n")
  f.write("        xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n")
  f.write("        width=\"2.2in\" height=\"1.6in\"\n")
  f.write("        viewBox=\"-10 -10 1010 1010\">\n")
  for a in range(N):
    f.write("<circle cx=\"%d\" cy=\"%d\" r=\"25\"\n" % \
        (int(1000*x[a]),int(1000*y[a])))
    f.write(" style=\"")
    if gene[a] == 1:
      f.write("fill:#ff0000;")
    else:
      f.write("fill:#00ff00;")
    f.write("stroke:#000000;stroke-width:8;\"/>\n")
    x0 = 0.0
    y0 = 0.0
    if gene[a] == 0:
      dmin = 1e100
      for b in range(N):
        if gene[b] == 1:
          #it is connected to a hub
          d = (x[a] - x[b])*(x[a] - x[b]) + (y[a] - y[b])*(y[a] - y[b])
          if d < dmin:
            dmin = d
            x0 = x[b]
            y0 = y[b]
    f.write("<polyline points=\"%d,%d " %
        (int(1000*x[a]),int(1000*y[a])))
    f.write("%d,%d\n" % (int(1000*x0),int(1000*y0)))
    f.write("\" style=\"stroke:#000000;stroke-width:8;\n")
    f.write("stroke-linejoin:miter stroke-linecap:butt;\"/>\n")
  f.write("</svg>\n")
  f.close()

# smaller the better
def fitness(x,y,gene,N):
  f = 0.0
  for a in range(N):
    # First if it is connected to the center at (0,0)
    if gene[a] == 1:
      f += x[a]*x[a] + y[a]*y[a]
    else:
      dmin = 1e100
      for b in range(N):
        if gene[b] == 1:
          # it is connected to a hub
          d = (x[a] - x[b])*(x[a] - x[b]) + (y[a] - y[b])*(y[a] - y[b])
          if d < dmin:
            dmin = d
      # if there are no hubs but we want to connect to them we get 1e100
      # as fitness which is bad so it's good :-)
      f += dmin
  return(f)

N = 10
x = np.random.random(10)
y = np.random.random(10)
gene = np.random.randint(2,size=N)
printsvg(x,y,gene,N,"a.svg")
print(fitness(x,y,gene,N))
