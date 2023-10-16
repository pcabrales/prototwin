#!/bin/bash
#
# mi_trabajo.sh
#
#SBATCH -J mi_trabajo
#SBATCH -p normal
#SBATCH --mem=3600
#SBATCH -o mitrabajo.%j.out
#SBATCH -e mitrabajo.%j.err
/bin/pwd
/bin/echo $SHELL
/bin/echo $PATH
/bin/echo ${LD_LIBRARY_PATH}
/bin/echo Estoy corriendo en el nodo `hostname`
/bin/echo Empiezo programa `date`
/home/vicvalla/topas/topas_victor Input.in
./single.x
/bin/echo Acabo programa `date`
