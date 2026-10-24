<div align="center">
<h1>Revisiting again Attention and Retention for Range View 3D LiDAR Semantic
Segmentation</h1>

<img
  src="https://github.com/ParvezAlam123/Revisiting-again-Attention-and-Retention-for-Range-View-3D-LiDAR-Semantic-Segmentation/blob/main/Network.png"
  alt="teaser"
  width="800"
/>

</div>

 LiDAR semantic segmentation is an important task for autonomous driving and robotics, enabling the detailed
understanding or surrounding environments for safe and reliable decision making. Range view representations provide the
computational efficiency by projecting 3D point cloud into 2D range images, making them suitable for real time perception.
However effectively capturing local geometric structures and long range contextual dependencies remains challenging,
under the computational constraints of real robotic systems. To address these challenges, we propose learnable weights
guided an Attention-Retention fusion block making RangeARNet point cloud segmentation network. The proposed method
integrates attention based feature modeling with retention based sequence processing to capture the complimentary
spatial and contextual information while maintaining the computational constraints. By exploiting the strength of both
mechanisms the segmentation network aims to improve feature representation and semantic discrimination across
diverse LiDAR scene structures. We evaluate the proposed approach on two public benchmarks, SemanticKITTI and
SemanticPOSS datasets, to assess its segmentation performance and generalization across different environments.
We got 59.3 % mIoU on SemanticKITTI test set and 53.06 % mIoU on SemanticPOSS test dataset. Furthermore
we conduct the real time experiments using data acquired from a physical LiDAR sensor to investigate the practical
feasibility fo the proposed architecture for real robotic perception. Experimental results demonstrates the effectiveness of
RangeARNet across benchmark evaluation and real-time sensor deployment for efficient LiDAR semantic segmentation.



## Installation

Create an environment and install the required packages with `pip install -r requirements.txt`.

In practice, you just need common packages as `torch`, `numpy`, `matplotlib`, `pyyaml`, `timm`, `tqdm`.

Code has been tested with the following versions of torch:

* torch 2.2.2 and CUDA 12.1
* torch 2.4.0 and CUDA 12.4

## Data Preparation

### SemanticKITTI

Download SemanticKITTI dataset from the [official website](https://semantic-kitti.org/).

### PandaSet

Download PandaSet dataset from the [official website](https://pandaset.org/) and raw Pandar64 LiDAR data from the link provided in this [Github Issue](https://github.com/scaleapi/pandaset-devkit/issues/67#issuecomment-674403708).

Follow the instructions on this [repo](https://github.com/SiMoM0/Pandaset2Kitti) to convert the PandaSet dataset to the SemanticKITTI format.

### SemanticPOSS

Download SemanticPOSS dataset from the [official website](http://www.poss.pku.edu.cn/semanticposs.html).


## Training

Run the following commands to train, specifying the dataset and log paths:

```shell
### SemanticKITTI
python train.py --dataset /path/to/semantickitti/ --data ./config/labels/semantic-kitti.yaml --config ./config/RangeRet-semantickitti.yaml --log ./log/kitti [--fp16]

### PandaSet
python train.py --dataset /path/to/pandaset/ --data ./config/labels/pandaset.yaml --config ./config/RangeRet-pandaset.yaml --log ./log/panda [--fp16]

### SemanticPOSS
python train.py --dataset /path/to/semanticposs/ --data ./config/labels/semantic-poss.yaml --config ./config/RangeRet-poss.yaml --log ./log/poss [--fp16]
```

## Testing

Run the following scripts to infer on a specific dataset:

```shell
### SemanticKITTI
python infer.py --dataset /path/to/semantickitti/ --data ./config/labels/semantic-kitti.yaml --config ./config/RangeRet-semantickitti.yaml --model /path/to/model.pt --split valid/test --log /path/to/predictions --fp16 [--save]

### PandaSet
python infer.py --dataset /path/to/pandaset/ --data ./config/labels/pandaset.yaml --config ./config/RangeRet-pandaset.yaml --model /path/to/model.pt --split valid/test --log /path/to/predictions --fp16 [--save]

### SemanticPOSS
python infer.py --dataset /path/to/semanticposs/ --data ./config/labels/semantic-poss.yaml --config ./config/RangeRet-poss.yaml --model /path/to/model.pt --split valid/test --log /path/to/predictions --fp16 [--save]

### E.g. for SemanticKITTI and pretrained model rangeret-kitti-657
# python3 infer.py --dataset /semanticKITTI/ --data ./config/labels/semantic-kitti.yaml --config config/RangeRet-semantickitti.yaml --model ./rangeret-kitti-657.pt --split valid --fp16 [--save] --log ./out/kitti_results
```



```


## Acknowledgment

Thanks to these great repositories: https://github.com/SiMoM0/RangeRet
