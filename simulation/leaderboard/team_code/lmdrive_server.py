"""
LMDrive Remote Inference Server
FastAPI service that loads the LMDrive model and serves inference requests.
"""

import os
import sys
import time
import base64
import importlib.util
import argparse
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms

sys.path.insert(0, 'simulation/assets/LMDrive/LAVIS')
sys.path.insert(0, 'simulation/assets/LMDrive/vision_encoder')
from lavis.common.registry import registry

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Resize2FixedSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pil_img):
        pil_img = pil_img.resize(self.size)
        return pil_img


def create_carla_rgb_transform(
    input_size, need_scale=True, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    tfl = []

    if isinstance(input_size, (tuple, list)):
        input_size_num = input_size[-1]
    else:
        input_size_num = input_size

    if need_scale:
        if input_size_num == 112:
            tfl.append(Resize2FixedSize((170, 128)))
        elif input_size_num == 128:
            tfl.append(Resize2FixedSize((195, 146)))
        elif input_size_num == 224:
            tfl.append(Resize2FixedSize((341, 256)))
        elif input_size_num == 256:
            tfl.append(Resize2FixedSize((288, 288)))
        else:
            raise ValueError("Can't find proper crop size")
    tfl.append(transforms.CenterCrop(img_size))
    tfl.append(transforms.ToTensor())
    tfl.append(transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)))

    return transforms.Compose(tfl)


class InferRequest(BaseModel):
    session_id: str = "default"
    ego_id: int = 0
    rgb_front: str
    rgb_left: str
    rgb_right: str
    rgb_rear: str
    lidar: str
    num_points: int = 35000
    target_point: List[float]
    velocity: float = 0.0
    text_input: str = "Follow the lane"
    reset_buffer: bool = False
    use_notice: bool = False
    notice_text: str = ""
    notice_frame_id: int = -1


class InferResponse(BaseModel):
    waypoints: List[List[float]]
    end_prob: float
    buffer_length: int
    inference_time_ms: float


class ResetRequest(BaseModel):
    session_id: str = "default"
    ego_id: int = 0


app = FastAPI(title="LMDrive Inference Server")

model = None
config = None
rgb_front_transform = None
rgb_left_transform = None
rgb_right_transform = None
rgb_center_transform = None
softmax_fn = None
visual_feature_buffer = defaultdict(list)
sample_rate = 4


def update_and_collect(image_embeds, session_id, ego_id):
    """Append frame embedding and return stacked temporal features."""
    global visual_feature_buffer, sample_rate
    key = (session_id, ego_id)
    visual_feature_buffer[key].append(image_embeds)
    result = visual_feature_buffer[key][::sample_rate]
    if (len(visual_feature_buffer[key]) - 1) % sample_rate != 0:
        result.append(visual_feature_buffer[key][-1])
    return torch.stack(result, 1)


def decode_image(b64_str):
    """Decode base64 JPEG string to PIL Image (RGB)."""
    raw = base64.b64decode(b64_str)
    return Image.open(BytesIO(raw)).convert("RGB")


def decode_lidar(b64_str):
    """Decode base64-encoded numpy float32 array of shape (40000, 4)."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(40000, 4).copy()
    return arr


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/reset")
def reset_buffer(req: ResetRequest):
    global visual_feature_buffer
    key = (req.session_id, req.ego_id)
    visual_feature_buffer[key] = []
    return {"status": "ok", "session_id": req.session_id, "ego_id": req.ego_id}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    global visual_feature_buffer

    t0 = time.time()

    buf_key = (req.session_id, req.ego_id)

    if req.reset_buffer:
        visual_feature_buffer[buf_key] = []

    pil_front = decode_image(req.rgb_front)
    pil_left = decode_image(req.rgb_left)
    pil_right = decode_image(req.rgb_right)
    pil_rear = decode_image(req.rgb_rear)

    with torch.no_grad():
        rgb_front = rgb_front_transform(pil_front).unsqueeze(0).cuda().float()
        rgb_left = rgb_left_transform(pil_left).unsqueeze(0).cuda().float()
        rgb_right = rgb_right_transform(pil_right).unsqueeze(0).cuda().float()
        rgb_rear = rgb_right_transform(pil_rear).unsqueeze(0).cuda().float()
        pil_center = pil_front.resize((800, 600))
        rgb_center = rgb_center_transform(pil_center).unsqueeze(0).cuda().float()

    lidar_np = decode_lidar(req.lidar)

    input_data = {}
    input_data["rgb_front"] = rgb_front
    input_data["rgb_left"] = rgb_left
    input_data["rgb_right"] = rgb_right
    input_data["rgb_center"] = rgb_center
    input_data["rgb_rear"] = rgb_rear
    input_data["target_point"] = torch.tensor(req.target_point).cuda().view(1, 2).float()
    input_data["lidar"] = torch.from_numpy(lidar_np).float().cuda().unsqueeze(0)
    input_data["num_points"] = torch.tensor([req.num_points]).cuda().unsqueeze(0)
    input_data["velocity"] = torch.tensor([req.velocity]).cuda().view(1, 1).float()
    input_data["text_input"] = [req.text_input]

    with torch.no_grad():
        image_embeds = model.visual_encoder(input_data)
        image_embeds = update_and_collect(image_embeds, req.session_id, req.ego_id)
        input_data["valid_frames"] = [image_embeds.size(1)]

        if req.use_notice:
            input_data["notice_text"] = [req.notice_text]
            input_data["notice_frame_id"] = [req.notice_frame_id]

        with torch.cuda.amp.autocast(enabled=True):
            waypoints, is_end = model(input_data, inference_mode=True, image_embeds=image_embeds)

    waypoints = waypoints[-1]
    waypoints = waypoints.view(5, 2)
    end_prob_val = softmax_fn(is_end)[-1][1].item()

    waypoints_list = waypoints.data.cpu().numpy().tolist()

    buf_len = len(visual_feature_buffer[buf_key])

    t1 = time.time()
    inference_ms = (t1 - t0) * 1000.0

    return InferResponse(
        waypoints=waypoints_list,
        end_prob=end_prob_val,
        buffer_length=buf_len,
        inference_time_ms=inference_ms,
    )


def load_model(config_path, device="cuda"):
    global model, config, rgb_front_transform, rgb_left_transform
    global rgb_right_transform, rgb_center_transform, softmax_fn, sample_rate

    spec = importlib.util.spec_from_file_location("LMDriverConfig", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.GlobalConfig()

    rgb_front_transform = create_carla_rgb_transform(224)
    rgb_left_transform = create_carla_rgb_transform(128)
    rgb_right_transform = create_carla_rgb_transform(128)
    rgb_center_transform = create_carla_rgb_transform(128, need_scale=False)

    sample_rate = config.sample_rate * 2

    model_cls = registry.get_model_class('vicuna_drive')
    print("[LMDrive Server] Building model...")
    net = model_cls(
        preception_model=config.preception_model,
        preception_model_ckpt=config.preception_model_ckpt,
        llm_model=config.llm_model,
        max_txt_len=64,
        use_notice_prompt=config.agent_use_notice,
    )

    print("[LMDrive Server] Loading checkpoint...")
    net.load_state_dict(torch.load(config.lmdrive_ckpt, weights_only=False)["model"], strict=False)
    net.to(device)
    net.eval()
    model = net
    softmax_fn = torch.nn.Softmax(dim=1)

    print("[LMDrive Server] Model loaded successfully.")

    print("[LMDrive Server] Running warm-up inference...")
    warmup_input = {}
    warmup_input["rgb_front"] = torch.randn(1, 3, 224, 224).cuda().float()
    warmup_input["rgb_left"] = torch.randn(1, 3, 128, 128).cuda().float()
    warmup_input["rgb_right"] = torch.randn(1, 3, 128, 128).cuda().float()
    warmup_input["rgb_center"] = torch.randn(1, 3, 128, 128).cuda().float()
    warmup_input["rgb_rear"] = torch.randn(1, 3, 128, 128).cuda().float()
    warmup_input["target_point"] = torch.zeros(1, 2).cuda().float()
    warmup_input["lidar"] = torch.zeros(1, 40000, 4).cuda().float()
    warmup_input["num_points"] = torch.tensor([[1000]]).cuda()
    warmup_input["velocity"] = torch.zeros(1, 1).cuda().float()
    warmup_input["text_input"] = ["Drive safely"]

    warmup_session = "__warmup__"
    warmup_ego_id = 999
    with torch.no_grad():
        image_embeds = model.visual_encoder(warmup_input)
        image_embeds = update_and_collect(image_embeds, warmup_session, warmup_ego_id)
        warmup_input["valid_frames"] = [image_embeds.size(1)]
        with torch.cuda.amp.autocast(enabled=True):
            model(warmup_input, inference_mode=True, image_embeds=image_embeds)

    visual_feature_buffer.pop((warmup_session, warmup_ego_id), None)
    print("[LMDrive Server] Warm-up complete.")


def main():
    parser = argparse.ArgumentParser(description="LMDrive Inference Server")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to lmdriver_config.py")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    load_model(args.config)

    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
