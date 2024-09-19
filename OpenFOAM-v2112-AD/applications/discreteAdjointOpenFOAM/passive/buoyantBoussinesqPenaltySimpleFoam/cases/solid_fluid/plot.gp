#! /usr/bin/gnuplot

set y2tics
plot \
     "postProcessing/sample/1000/x0.012_U.xy"           u 1:2         w l t "U solid fluid", \
     "postProcessing/sample/1000/x0.012_T.xy"           u 1:2 ax x1y2 w l t "T solid fluid"

pause -1
