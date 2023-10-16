
set auto 

fcal1='calibration-det1.txt'

p fcal1 u 2:1

r1(x)=a1*x**2+b1*x+c1

#r1(x)=b1*x+c1

fit r1(x) fcal1 u 2:1 via a1,b1,c1

p fcal1 u 2:1, r1(x) w l  

#set print 'calespect.txt' 
print c1*344/405.,b1*344/405.,a1*344/405.
