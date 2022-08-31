Place the Evaluation script in the same directory as the detection_image script(which will contain the functionality to generate the model.csv file containing prediction data of box coordinates,label name, percentage probability and return the inference time for each image passed from the main script).
Make an empty folder named output_files in the same directory to save the output files(class wise) generated after evaluation.

Evaluation script changes:
Appropriate paths were set to take as input the test images and test txt files as annotations in the main() function.
Class wise dictionaries(containing bounding box data for GT and predicted values) were prepared from txt files(for GT) and generated csv files(for predictions). They were passed as input to the find_average function(line 150) for calculating iou average and other parameters for each class. Results would be stored in the output_files directory.

detection_image script changes:
This would be the script for generating the model output(presumably bounding box coordinates, confidence scores and labels) after passing it an image.
Inference time was calculated(considering part of the code returning outputs from pretrained model, line 47 in attached script) and returned inside the Evaluation script by calling detection_on_image function. It would be saved in the latency.csv file in output_files directory later.

For each class, corresponding iou and count csv files will be generated along with a latency.csv(containing inference time for each image) and a model output csv file(retina_r.csv) will be stored inside the output_files folder.  




  
