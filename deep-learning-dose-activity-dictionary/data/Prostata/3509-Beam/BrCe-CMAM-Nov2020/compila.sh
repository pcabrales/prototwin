# gcc  -O3 -I/opt/picoscope/include  -Wl,-u,pthread_atfork -lps6000 -lpthread -L.   $1  -o $1.x
#gcc  -O3   -u,pthread_atfork -lps6000 -lpthread -L. -I.    $1  -o $1.x
#gcc -I. -lps6000  -Ofast -mcmodel=medium -u,pthread_atfork  -lpthread   $1  -o $1.x
#gcc -O1 -I. $1 -lps6000 -mcmodel=medium -u,pthread_atfork -lpthread -fbounds-check -o ps6000.x
#gcc -O1 -I. $1 -lps6000 -mcmodel=medium -u,pthread_atfork -lpthread -o ps6000.x
gcc -g -O3 -I. -L/opt/picoscope/lib/ $1 -lps6000 -mcmodel=medium -u,pthread_atfork -lpthread -o ps6000.x
