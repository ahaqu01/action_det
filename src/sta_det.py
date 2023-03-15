import collections
import numpy as np
import torch
from .utils.get_cfgs import get_cfg
from .utils.geometric import imresize, rescale_size, imnormalize
from .models.sta_model import sta_model


class StdetPredictor(object):
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        model_cfg (str): Path to stdet config.
        model_weights (str): Path to stdet checkpoint.
        stdet_input_shortside (int): input img shortest side
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self,
                 model_cfg="/workspace/huangniu_demo/action_det/src/configs/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_infer.py",
                 model_weights="/workspace/huangniu_demo/action_det/src/weights/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth",
                 stdet_input_shortside=256,
                 device=None,
                 score_thres=0.4,
                 label_map_path="/workspace/huangniu_demo/action_det/src/configs/label_map.txt"):

        self.device = device
        self.stdet_input_shortside = stdet_input_shortside
        self.score_thr = score_thres

        # get configs
        self.cfg = get_cfg(model_cfg)

        # create model & load weights
        self.model = sta_model(self.cfg)
        weights_state_dict = torch.load(model_weights)["state_dict"]
        model_state_dict = collections.OrderedDict()
        for n_p_1, n_p_2 in zip(self.model.state_dict().items(), weights_state_dict.items()):
            if n_p_1[1].shape == n_p_2[1].shape:
                model_state_dict[n_p_1[0]] = n_p_2[1]
        self.model.load_state_dict(model_state_dict, strict=True)
        self.model.to(device)
        self.model.eval()

        # init label map, aka class_id to class_name dict
        with open(label_map_path) as f:
            lines = f.readlines()
        lines = [x.strip().split(': ') for x in lines]
        self.label_map = {int(x[0]): x[1] for x in lines}

    def pre_process(self, frames, bboxs):
        # process frames
        processed_frames = []
        h, w = frames[0].shape[:2]
        stdet_input_size = rescale_size((w, h), (self.stdet_input_shortside, np.Inf))
        for frame in frames:
            processed_frame = imresize(frame, stdet_input_size).astype(np.float32)
            imnormalize(processed_frame,
                        mean=self.cfg.img_norm_cfg["mean"],
                        std=self.cfg.img_norm_cfg["std"],
                        to_rgb=self.cfg.img_norm_cfg["to_bgr"])
            processed_frames.append(processed_frame)
        # get frames tensor
        input_array = np.stack(processed_frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(self.device)

        ratio = tuple(n / o for n, o in zip(stdet_input_size, (w, h)))
        st_bboxs = bboxs.clone()
        st_bboxs[:, ::2] = st_bboxs[:, ::2] * ratio[0]
        st_bboxs[:, 1::2] = st_bboxs[:, 1::2] * ratio[1]
        st_bboxs = torch.cat([torch.zeros((st_bboxs.shape[0], 1)).to(self.device), st_bboxs[:, :4]],
                             dim=-1)

        return dict(
            img=input_tensor,
            proposals=st_bboxs)

    def post_process(self, result, bboxs):
        # pack results of human detector and stdet
        preds = {}
        for bbox_id in range(bboxs.shape[0]):
            preds[bbox_id] = []
            for class_id in range(result.shape[-1]):
                if class_id not in self.label_map:
                    continue
                if result[bbox_id, class_id].item() > self.score_thr:
                    preds[bbox_id].append([self.label_map[class_id], result[bbox_id][class_id].item()])

        return preds

    def inference_single_clip(self, frames, bboxs):
        """Spatio-temporval Action Detection model inference."""

        # pre process data
        input_data = self.pre_process(frames, bboxs)

        # model infer
        with torch.no_grad():
            result = self.model(**input_data)

        # pack results of human detector and stdet
        preds = self.post_process(result, bboxs)
        return preds


if __name__ == "__main__":
    st_p = StdetPredictor()
    print(st_p)
