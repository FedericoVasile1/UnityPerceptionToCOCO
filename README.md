# Convert from [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) to COCO labeling

Unity Perception <b><= 0.11.2</b> stores labels and metadata in sequential `capture_*.json` files according to a certain format. One sample output is the following: 
```
Perception_dataset_base_folder   
|- Dataset*
|   |- captures_000.json
|   |- captures_001.json
|   |- captures_*.json
|   |- annotation_definitions.json
|   |- metric_definitions.json
|   |- sensors.json
|
|- RGB*
|   |- rgb_2.png
|   |- rgb_3.png
|   |- rgb_*.png
|
|- InstanceSegmentation*
|   |- segmentation_2.png
|   |- segmentation_3.png
|   |- segmentation_*.png
```
I provide a script to extract a `coco.json` file in COCO format. The conversion from segmentation mask images to `"segmentation"` field in the COCO format is based on https://github.com/chrise96/image-to-coco-json-converter

##  Install
Python 3 is required.
```bash
pip install scikit-image
pip install shapely
pip install matplotlib
pip install pycocotools
```

## Usage 
Specify the base folder of the dataset and the name of the generated COCO file:
```bash
python convert.py --input_dataset_path YOUR_Perception_dataset_base_folder --output_coco_file YOUR_Perception_dataset_base_folder/coco.json
```
Use [pycocotools](https://github.com/cocodataset/cocoapi) to read the COCO file and visualize some images:
```bash
python visualize.py --images_dir YOUR_Perception_base_folder/RGB_string --coco_file YOUR_Perception_dataset_base_folder/coco.json --num_imgs_to_show 10 --show_bbox --show_text_labels
```