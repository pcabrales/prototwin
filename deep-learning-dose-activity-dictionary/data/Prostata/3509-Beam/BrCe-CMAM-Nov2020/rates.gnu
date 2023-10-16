

set xlabel 's'
set ylabel 'cps'

set auto

p 'rates-det1.txt' w lp t 'Det A'
rep 'rates-det2.txt' w lp t 'Det B'
rep 'rates-det3.txt' w lp t 'Det C'
rep 'rates-det4.txt' w lp t 'Det D'
