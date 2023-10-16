
set grid
set xrange [0:2200]

set xlabel 'keV'
set ylabel 'Counts'

p 'ener.txt' u 1:2 w step t 'det A'
rep 'ener.txt' u 1:3 w step t 'det B'
rep 'ener.txt' u 1:4 w step t 'det C'
rep 'ener.txt' u 1:5 w step t 'det D'
