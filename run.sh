classifier=2 
parallel=0
validation="false"
visualiseFrames="false"
visualisesubFrames="false"
verbose="false"
savefile="false"

Help()
{
   echo
   echo "Run this script with -r flag to execute all approaches with results validation enabled"
   echo
   echo "Syntax: Template [-c|p|t|f|s|v]"
   echo
   echo "options:"
   echo
   echo "c     0 = Slow SVM, 1 = Fast SVM, 2 = Modified YOLO, 3 = Orginal YOLO"
   echo "p     0 = For normal processing, 1 = For parallel processing"
   echo "t     Validate/test results"
   echo "f     Visualize/Display Frames"
   echo "s     Print software version and exit."
   echo "v     Verbose mode."
   echo "k     Save/keep as file."
   echo
   echo "Default Option values -c 2 -p 0 -t False -f False -s False -v False"
   echo 
}

ExecAll()
{

c=0
x=0
p=0
while [ $c -lt 4 ]
do
   if [ $c -lt 2 ]
   then
      p=$(($x+0))
   	while [ $p -lt 2 ]
	   do
         if [ $c -eq 0 -a $p -eq 0 ]; then echo '===============================Model A without parallel==============================='
         elif [ $c -eq 0 -a $p -eq 1 ]; then  echo '================================Model A with parallel================================='
         elif [ $c -eq 1 -a $p -eq 0 ]; then  echo '===============================Model B without parallel==============================='
         else 
            echo '================================Model B with parallel================================='
         fi
		   python3 detectionsystem.py -c=$c -p=$p -t="true"
         p=`expr $p + 1`
      done 
   else 
   	if [ $c -eq 2 ]; then echo '====================================Modified YOLOv3===================================='
        else  echo '====================================Original YOLOv3===================================='
   	fi
   	python3 detectionsystem.py -c=$c -p=0 -t="true"
   fi
   c=`expr $c + 1`
done


}
while getopts ":hrc:p:t:f:s:v:k:" option; do
  case $option in
    h ) 
    Help 
    exit;;
    r )
    ExecAll
    exit;;
    c ) classifier=${OPTARG:-2}
    ;;
    p ) parallel=${OPTARG:-0}
    ;;
    t ) validation=${OPTARG:-False} 
    ;;
    f ) visualiseFrames=${OPTARG:-False} 
    ;;
    s ) visualisesubFrames=${OPTARG:-False} 
    ;;
    v ) verbose=${OPTARG:-False} 
    ;;
    k ) savefile=${OPTARG:-False} 
    ;;
  esac
done

PrintValues()
{
   echo classifier 
   echo $classifier 
   echo parallel
   echo $parallel
   echo validation
   echo $validation
   echo visualiseFrames
   echo $visualiseFrames
   echo visualisesubFrames
   echo $visualisesubFrames
   echo verbose
   echo $verbose
   echo savefile
   echo $savefile
}
#PrintValues

python3 detectionsystem.py -c $classifier -p $parallel -t $validation -f $visualiseFrames -s $visualisesubFrames -v $verbose -k $savefile
