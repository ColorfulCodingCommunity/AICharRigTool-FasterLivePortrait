# python cc_webcam2pic-control.py --src_image assets/examples/source/s40.jpg --cfg configs/trt_infer.yaml --realtime
# python client.py

import os
import argparse
import cv2
import time
import numpy as np
import os
from omegaconf import OmegaConf

import asyncio
import websockets
import base64

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

idx = 0
pipe = None

def process_image(image_data):

    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"failed to read driving image! exit!")
        exit(1)

    t0 = time.time()

    global idx
    isFirstFrame = idx == 0

    dri_crop, out_crop, out_org = pipe.run(frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=isFirstFrame)
    if out_crop is None:
        print("no face in driving image!")
        exit(1)

    infer_time = time.time() - t0
    print(f"[{idx}]Inference time: {infer_time} seconds, {isFirstFrame}")
    idx += 1

    dri_crop = cv2.resize(dri_crop, (512, 512))
    out_crop = np.concatenate([dri_crop, out_crop], axis=1)
    out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)

    output_path = "./results/output.jpg"
    if infer_cfg.infer_params.flag_pasteback:
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        print("inference median time: {} ms/frame".format(infer_time * 1000))
        cv2.imwrite(output_path, out_org)
        return out_org
    else:
        cv2.imwrite(output_path, out_crop)
        print("inference median time: {} ms/frame".format(infer_time * 1000))
        return out_crop



async def handle_client(websocket, path):
    async for message in websocket:
        print("Received image data")
        # Process the image
        processed_image = process_image(message)

        _, buffer = cv2.imencode('.jpg', processed_image)
        base64_response = base64.b64encode(buffer).decode('utf-8')
        # Send back the processed image
        await websocket.send(base64_response)
        print("Sent processed grayscale image")

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9870, ping_timeout=None, ping_interval=None):
        print("Server started and waiting for connections...")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='source image')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    args, unknown = parser.parse_known_args()

    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = False ## Default False

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal= False)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)

    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)

    asyncio.run(main())