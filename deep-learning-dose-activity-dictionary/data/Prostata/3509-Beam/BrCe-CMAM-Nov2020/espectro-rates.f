        program espectro

        implicit real(a-h,o-z)
        integer*2 ikk,ikkk
        real ene,tesp,e1,e2
        real*8 time,tbase(4),tbefore(4),trate
        real  rates(4),ratesw(4),t
        integer nevent(4),neventw(4),nespec(4,2500),enerdel(4,2500)
        real a(4),b(4),c(4)
        real maxi,maxi2,ymax,sumn
        integer(kind=8) i

        character*40 inputfile

        ncont=0
	nncc=0


c       Program to generate the rates in the four detectors
        ntt=1
        ikk=0

        call getarg (1,inputfile)

        open(12,file="input-cal.txt")       
        open(13,file=inputfile,form='unformatted', access='stream')
        open(14,file='rates-det1.txt')
        open(15,file='rates-det2.txt')
        open(16,file='rates-det3.txt')
        open(17,file='rates-det4.txt')
        open(18,file='ener.txt')
        open(98,file='enerw.txt')
        open(21,file='rates-det1w.txt')
        open(22,file='rates-det2w.txt')
        open(23,file='rates-det3w.txt')
        open(24,file='rates-det4w.txt')

        read(12,*)c(1),b(1),a(1)
        read(12,*)c(2),b(2),a(2)
        read(12,*)c(3),b(3),a(3)
        read(12,*)c(4),b(4),a(4)
        read(12,*)step
        read(12,*)t
        read(12,*)e1,e2
        read(12,*)t1,t2
        tbase=t
        print*,t1,t2

        print*,'Leyendo fichero.txt'
        enerdel=0
        nevent=0
        tbefore=0

        do i=1,1000000000

c         read(13,end=231)ikk,time,sumn,maxi2,ymax
         read(13,end=231)ikk,time,ymax,ikkk
         ene=c(ikk+1)+ymax*b(ikk+1)+a(ikk+1)*ymax**2
c         ene=c(ikk+1)+sumn*b(ikk+1)+a(ikk+1)*sumn**2
         
c         if(ikk.eq.0)write(99,*),ikk,time,ene
         if(ene.gt.0..and.ene.lt.2500)then
          nevent(ikk+1)=nevent(ikk+1)+1
          if(ene.gt.e1.and.ene.lt.e2)neventw(ikk+1)=neventw(ikk+1)+1
          if(time.gt.tbase(ikk+1))then
           trate=time-tbefore(ikk+1)
           rates(ikk+1)=(nevent(ikk+1)-1)/trate
           ratesw(ikk+1)=(neventw(ikk+1)-1)/trate
           nevent(ikk+1)=1
           neventw(ikk+1)=1
           write(14+ikk,*)tbase(ikk+1),rates(ikk+1),time
           write(21+ikk,*)tbase(ikk+1),ratesw(ikk+1),time
           tbase(ikk+1)=tbase(ikk+1)+t
           tbefore(ikk+1)=time
          endif
         
          inn=min(int(ene/step)+1,500)
          nespec(ikk+1,inn)=nespec(ikk+1,inn)+1
        if (time.gt.t1.and.time.lt.t2) then
        enerdel(ikk+1,inn)=enerdel(ikk+1,inn)+1
        endif
         endif

        enddo

  231   continue

        close(13)
        do is=1,500
         write(18,*)is*step,(nespec(ic,is),ic=1,4)
         write(98,*)is*step,(enerdel(ic,is),ic=1,4)
        enddo
        print*,i,time
        end program espectro
        
