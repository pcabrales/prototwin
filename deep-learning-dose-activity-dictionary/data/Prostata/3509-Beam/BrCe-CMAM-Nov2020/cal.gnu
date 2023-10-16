g(x)=A*exp(-(x-m)**2/(2*s**2))+B*x+C
unset print

p 'ener.txt' u 1:2 w step

pause mouse

x1=MOUSE_X

pause mouse

x2=MOUSE_X

set xrange [x1:x2]

rep

pause mouse

m=MOUSE_X
A=MOUSE_Y
s=m*0.1

fit g(x) 'ener.txt' u 1:2  via A,m,s,B,C

prin 2.35*s/m
set print 'calibration-det1.txt' append

print e,m

rep g(x)
