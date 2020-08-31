

downloadfiles()
{
if [ -e data/YOLOv3/yolov3.weights ]
then
    echo "yolov3.weights present"
else
    #wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=19f64_-Kfv-zJZT15blxqXMVw3oA4uM1Z' -O data/YOLOv3/yolov3.weights  -q --show-progress
    wget 'https://pjreddie.com/media/files/yolov3.weights' -O data/YOLOv3/yolov3.weights  -q --show-progress
   
fi

if [ -e data/YOLOv3/yolov3_classes.txt  ]
then
    echo "yolov3_classes.txt present"
else
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1yo1mcj5nlpctXTcXJwrT649-IvNVqAsT' -O data/YOLOv3/yolov3_classes.txt  -q --show-progress
   
fi

if [ -e data/YOLOv3/yolov3.cfg ]
then
    echo "yolov3.cfg present"
else
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1S8j7_2TtpghtLK1kKurbUpi-QHT_vBgW' -O data/YOLOv3/yolov3.cfg -q --show-progress
fi
}
if [[ ! -e data ]]; then
     mkdir -p data/YOLOv3
	downloadfiles
else
   if [[ ! -e data/YOLOv3 ]]; then
     mkdir  data/YOLOv3
   fi
   downloadfiles
fi

if [[ ! -e data/v1.0-mini ]]; then
	echo
	echo "Please download the nuscenes dataset from https://www.nuscenes.org/ . If you are professor Derek Molloy, I have shared the dataset with you personally through GDrive. You can download it from there too. :)"
	echo
	exit;
fi
pip3 install -r requirements.txt
