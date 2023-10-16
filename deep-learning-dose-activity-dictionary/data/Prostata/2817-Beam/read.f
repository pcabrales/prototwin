        real dose(150,60,70)



        open(12,file='Dose.raw',access='stream',status='old')
        read(12)dose

        print*, maxval(dose)

        end
