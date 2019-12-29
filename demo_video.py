import time
import sys
import os
import cv2
import argparse
import numpy as np
from util import COLORS_10, draw_bboxes

from deep_sort import DeepSort

# CenterNet
CENTERNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "CenterNet/src/lib/")
if os.path.isdir(CENTERNET_PATH):
    sys.path.insert(0, CENTERNET_PATH)
    from detectors.detector_factory import detector_factory
    from opts import opts
else:
    print(CENTERNET_PATH)
    exit()


class Detector(object):
    def __init__(self, centernet_opt, args):
        # CenterNet detector
        self.detector = detector_factory[centernet_opt.task](centernet_opt)
        # Deep SORT
        self.deepsort = DeepSort(args.deepsort_checkpoint,
                                 args.max_cosine_distance, args.use_cuda)
        self.args = args

    def run(self, video_path, output_path):
        # open input video
        assert os.path.isfile(video_path), "Error: invalid video path"
        vdo = cv2.VideoCapture()
        vdo.open(video_path)
        # open output video
        im_width = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output_vdo = cv2.VideoWriter(output_path, fourcc, 20, (im_width, im_height))
        # track each frame in video
        start_time = time.time()
        frame_cnt = 0
        while vdo.grab():
            frame_cnt += 1
            _, ori_im = vdo.retrieve()
            im = ori_im[0:im_height, 0:im_width]
            detection = self.detector.run(im)["results"][1]
            bbox_xywh, conf = Detector._bbox_to_xywh_cls_conf(detection, self.args.min_confidence)
            outputs = self.deepsort.update(bbox_xywh, conf, im)
            if(len(outputs) > 0):
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
            elapsed_time = time.time() - start_time
            print("Frame {:05d}, Time {:.3f}s, FPS {:.3f}".format(
                frame_cnt, elapsed_time, frame_cnt / elapsed_time))
            output_vdo.write(ori_im)

    @staticmethod
    def _bbox_to_xywh_cls_conf(bbox, min_confidence):
        bbox = bbox[bbox[:, 4] > min_confidence, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
        bbox[:, 0] = bbox[:, 0] + bbox[:, 2] / 2
        bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
        return bbox[:, :4], bbox[:, 4]


def parse_args():
    parser = argparse.ArgumentParser("CenterNet + DeepSORT, for video demo")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--model_path", help="Path to the model for CenterNet",
                        default="CenterNet/models/ctdet_coco_dla_2x.pth")
    parser.add_argument("--arch", help="CenterNet arch", default="dla_34")
    parser.add_argument("--deepsort_checkpoint", help="Checkpoint for deep SORT",
                        default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument(
        "--output", help="Path to the output video file",
        default="demo.avi")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument("--no_cuda", dest="use_cuda", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    centernet_opt = opts().init("ctdet --load_model {} --arch {} --gpus {}".format(
        args.model_path, args.arch, "0" if args.use_cuda else "-1").split(" "))
    centernet_opt.input_type = "vid"
    centernet_opt.vid_path = args.video_path
    det = Detector(centernet_opt, args)
    det.run(args.video_path, args.output)
