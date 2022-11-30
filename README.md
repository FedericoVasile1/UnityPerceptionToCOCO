# Convert from [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) to COCO format

Unity Perception <b><= 0.11.2</b> stores labels and metadata in sequential `captures_*.json` files according to a certain format. One sample output is the following: 
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
Given the structure above, I provide a script to extract a `coco.json` file in COCO format.

##  Install
Python 3 is required.
```bash
pip install matplotlib
pip install pycocotools
```
The converter works with datasets generated using Unity Perception <b><= 0.11.2</b>. The dataset must be generated with the following three labelers: [InstanceSegmentationLabeler](https://github.com/Unity-Technologies/com.unity.perception/tree/f45895f7dcad27dee545d6165a2f6c237554600a/com.unity.perception/Runtime/GroundTruth/Labelers/InstanceSegmentation), [BoundingBox2DLabeler](https://github.com/Unity-Technologies/com.unity.perception/tree/f45895f7dcad27dee545d6165a2f6c237554600a/com.unity.perception/Runtime/GroundTruth/Labelers/BoundingBox) [RenderedObjectInfoLabeler](https://github.com/Unity-Technologies/com.unity.perception/blob/f45895f7dcad27dee545d6165a2f6c237554600a/com.unity.perception/Runtime/GroundTruth/Utilities/RenderedObjectInfo.cs) as shown in the figure above (Camera Labelers of the [PerceptionCamera](https://github.com/Unity-Technologies/com.unity.perception/blob/f45895f7dcad27dee545d6165a2f6c237554600a/com.unity.perception/Runtime/GroundTruth/PerceptionCamera.cs)):

![annotators](https://user-images.githubusercontent.com/50639319/204907323-d51eb677-4623-431d-b195-c5a366a50c4f.png)

Make sure that the `Annotation Id` is the same as in figure.

## Usage 
Specify the base folder of the dataset a name for the COCO file to generate:
```bash
python convert.py --input_dataset_path YOUR_Perception_dataset_base_folder --output_coco_file YOUR_Perception_dataset_base_folder/coco.json
```
Use [pycocotools](https://github.com/cocodataset/cocoapi) to read the COCO file and visualize some images:
```bash
python visualize.py --images_dir YOUR_Perception_base_folder/RGB_base_folder --coco_file YOUR_Perception_dataset_base_folder/coco.json --num_imgs_to_show 10 --show_bbox --show_text_labels --random
```
