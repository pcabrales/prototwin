
RUN_NUMBER=1
DIR=RUN$RUN_NUMBER
echo $DIR
while test -d $DIR
 do
  RUN_NUMBER=$((RUN_NUMBER+1))
  DIR=RUN$RUN_NUMBER
 done
mkdir $DIR

dt=$(date)
echo $dt > $DIR/horaStart.txt

echo "Generate FIFO"
./ps6000.x

sleep 1

cp c0.txt $DIR
cp test.txt $DIR
cp test2.txt $DIR
cp out.stream $DIR
