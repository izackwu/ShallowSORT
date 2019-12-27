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


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


class Detector(object):
    def __init__(self, centernet_opt, args):
        # CenterNet detector
        self.detector = detector_factory[centernet_opt.task](centernet_opt)
        # Deep SORT
        self.deepsort = DeepSort(args.deepsort_checkpoint,
                                 args.max_cosine_distance, args.use_cuda)
        self.debug = args.debug
        if self.debug and not os.path.exists(args.debug_dir):
            os.mkdir(args.debug_dir)
        self.args = args

    def run(self, sequence_dir, output_file):
        assert os.path.isdir(sequence_dir), "Invalid sequence dir: {}".format(sequence_dir)
        seq_info = gather_sequence_info(sequence_dir, None)
        print("Start to handle sequence: {} (image size: {}, frame {} - {})".format(
            seq_info["sequence_name"], seq_info["image_size"], seq_info["min_frame_idx"],
            seq_info["max_frame_idx"]))
        start_time = time.time()
        frame_cnt = 0
        results = []
        for frame in range(seq_info["min_frame_idx"], seq_info["max_frame_idx"] + 1):
            frame_image = seq_info["image_filenames"][frame]
            frame_cnt += 1
            image = cv2.imread(frame_image)
            detection_result = self.detector.run(frame_image)["results"][1]
            xywh, conf = Detector._bbox_to_xywh_cls_conf(detection_result, self.args.min_confidence)
            output = self.deepsort.update(xywh, conf, image)
            for x1, y1, x2, y2, track_id in output:
                results.append((
                    frame, track_id, x1, y1, x2 - x1, y2 - y1  # tlwh
                ))
            elapsed_time = time.time() - start_time
            print("Frame {:05d}, Time {:.3f}s, FPS {:.3f}".format(
                frame_cnt, elapsed_time, frame_cnt / elapsed_time))
            if self.debug:
                detect_xyxy = detection_result[detection_result[:, 4] > self.args.min_confidence, :4]
                detect_image = draw_bboxes(image, detect_xyxy)
                cv2.imwrite(os.path.join(self.args.debug_dir,
                                         "{}-{:05}-detect.jpg".format(seq_info["sequence_name"], frame)), detect_image)
                if len(output) == 0:
                    continue
                image = cv2.imread(frame_image)
                track_image = draw_bboxes(image, output[:, :4], output[:, -1])
                cv2.imwrite(os.path.join(self.args.debug_dir,
                                         "{}-{:05}-track.jpg".format(seq_info["sequence_name"], frame)), track_image)

        print("Done. Now write output to {}".format(args.output_file))
        with open(output_file, mode="w") as f:
            for row in results:
                f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n" % (
                    row[0], row[1], row[2], row[3], row[4], row[5]))

    @staticmethod
    def _bbox_to_xywh_cls_conf(bbox, min_confidence):
        bbox = bbox[bbox[:, 4] > min_confidence, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]
        bbox[:, 0] = bbox[:, 0] + bbox[:, 2] / 2
        bbox[:, 1] = bbox[:, 1] + bbox[:, 3] / 2
        return bbox[:, :4], bbox[:, 4]


def parse_args():
    parser = argparse.ArgumentParser("CenterNet + DeepSORT, for MOTChallenge")
    parser.add_argument("sequence_dir", help="Path to MOTChallenge sequence directory")
    parser.add_argument("--model_path", help="Path to the model for CenterNet",
                        default="CenterNet/models/ctdet_coco_dla_2x.pth")
    parser.add_argument("--arch", help="CenterNet arch", default="dla_34")
    parser.add_argument("--deepsort_checkpoint", help="Checkpoint for deep SORT",
                        default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="output.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--debug_dir", help="Debugging info output directory", default="./debug")
    parser.add_argument("--no_cuda", dest="use_cuda", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    centernet_opt = opts().init("ctdet --load_model {} --arch {} --gpus {}".format(
        args.model_path, args.arch, "0" if args.use_cuda else "-1").split(" "))
    print(centernet_opt)
    det = Detector(centernet_opt, args)
    det.run(args.sequence_dir, args.output_file)
