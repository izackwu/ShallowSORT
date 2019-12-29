# ShallowSORT

SJTU EI339 Artificial Intelligence course project: Multi-Object Tracking based on Deep SORT.

## Demo

For example, the tracking result of [`MOT16-02`](https://motchallenge.net/vis/MOT16-02) is

<iframe width="560" height="315" src="https://www.youtube.com/embed/0Ba0rT8mfAQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Prerequisites

**Anaconda (or Miniconda) is highly recommended for this project.**

-   Install CenterNet according to [its instructions](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md)
-   Install necessary packages for deep SORT with `pip install -r requirements.txt`
-   Get pretrained models:
    -   CenterNet's models can be found in [the model zoo](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md), and they are supposed to be put under `CenterNet/models`
    -   Deep SORT's models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1WxaD7NHamVm8iR4Agfuoyzdh8bR4ET1T?usp=sharing) and then put under `deep_sort/deep/checkpoint`

## Usage

### MOT Challenge

`mot_challenge.py` takes the path to a MOT Challenge sequence and produces the tracking result in a text file (its format is consistent with MOT Challenge requirements for evaluation).
```bash
python mot_challenge.py [-h]
                          [--model_path MODEL_PATH]
                          [--arch ARCH]
                          [--deepsort_checkpoint DEEPSORT_CHECKPOINT]
                          [--output_file OUTPUT_FILE]
                          [--min_confidence MIN_CONFIDENCE]
                          [--min_detection_height MIN_DETECTION_HEIGHT]
                          [--nms_max_overlap NMS_MAX_OVERLAP]
                          [--max_cosine_distance MAX_COSINE_DISTANCE]
                          [--debug]
                          [--debug_dir DEBUG_DIR]
                          [--no_cuda]
                        sequence_dir
```

### Video Demo

`demo_video.py` takes the path to a video and produces the tracking result in a video for visualization.
```bash
python demo_video.py [-h]
                       [--model_path MODEL_PATH]
                       [--arch ARCH]
                       [--deepsort_checkpoint DEEPSORT_CHECKPOINT]
                       [--output OUTPUT]
                       [--min_confidence MIN_CONFIDENCE]
                       [--max_cosine_distance MAX_COSINE_DISTANCE]
                       [--no_cuda]
                    video_path

```


## References

-   SORT: Simple Online and Realtime Tracking, [paper](https://arxiv.org/pdf/1602.00763.pdf), [code](https://github.com/abewley/sort)
-   Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric, [paper](https://arxiv.org/pdf/1703.07402.pdf), [code](https://github.com/nwojke/deep_sort)
-   CenterNet: Objects as Points, [paper](https://arxiv.org/pdf/1904.07850.pdf), [code](https://github.com/xingyizhou/CenterNet)
-   Deep SORT with YOLOv3: [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
-   MOTChallenge: The Multiple Object Tracking Benchmark, [MOTChallenge](https://motchallenge.net)
