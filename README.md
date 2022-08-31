Dataset Preparation
	The base data directory should consist of the following files:
		"xml" folder - containing the annotations for IDD dataset in xml format
		"images" folder - containing the images for training
		"csv" folder - empty directory to store individual csv files
		"class.csv" - CSV file with Class mapping(Input to the Model)

		Running csv_gen.py at the base data directory:
		"train.csv" - generate CSV file from dataset(Input to the Model)

		Running csv_gen.py for generating test data:
		"test.csv" - generate CSV file from dataset(Input to the Model)

		
TRAIN 

	- \keras_retinanet\bin\train.py --weights snapshot\resnet50_coco_best_v2.1.0.h5 --compute-val-loss  --weighted-average --image-min-side 800 --image-max-side 800 --batch-size 4 --steps 7890 --epochs 70 csv data\train.csv data\class.csv 

	saved models
	- \keras_retinanet\bin\snapshots\


Evaluate : 

	- \tools\test_lanenet.py --weights_path E:\Abhishek\Lane_Detection\CULane\parth\LaneNet\lanenet-lane-detection-master\model\tusimple\bisenetv2_lanenet\tusimple_train_miou=0.4496.ckpt-24  --image_path ./data/image/201_frame2819_leftImg8bit.jpg

	- \keras_retinanet\bin\evaluate.py --gpu 0 --image-min-side 800 --image-max-side 800 --save-path E:\IISc\Object_detection\keras-retinanet\keras-retinanet-main\output_images csv E:\IISc\Object_detection\keras-retinanet\keras-retinanet-main\data\train_short.csv E:\IISc\Object_detection\keras-retinanet\keras-retinanet-main\data\class.csv E:\IISc\Object_detection\keras-retinanet\keras-retinanet-main\keras_retinanet\bin\snapshots\inference_converted\resnet50_csv_47_converted.h5



	- EVALUATE ON CUSTOM TEST DATASET :
		\tool \evaluate_lanenet_on_tusimple.py --image_dir E:\\Abhishek\\Lane_Detection\\CULane\\parth\\LaneNet\\lanenet-lane-detection-master\\data\\test_set --weights_path E:\Abhishek\Lane_Detection\CULane\parth\LaneNet\lanenet-lane-detection-master\model\tusimple\bisenetv2_lanenet\tusimple_train_miou=0.5029.ckpt-64 --save_dir E:\\Abhishek\\Lane_Detection\\CULane\\parth\\LaneNet\\lanenet-lane-detection-master\\data\\test_set\\test_output (keep omne category of test data at a time)


EVALUATION

	- \data\Evaluate\
		- Generate csv files with results
