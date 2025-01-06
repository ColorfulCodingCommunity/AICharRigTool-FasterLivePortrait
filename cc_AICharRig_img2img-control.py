# python run2.py --src_image assets/examples/source/s40.jpg --cfg configs/trt_infer.yaml --realtime
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

sourceImage = None

def process_image(driving_image_data):
    global idx
    # nparr = np.frombuffer(base64.b64decode(driving_image_data), np.uint8)
    nparr = np.frombuffer(driving_image_data, np.uint8)
    driving_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if driving_frame is None:
        print(f"failed to read driving image! exit!")
        return None
    
    isFirstFrame = idx == 0

    ret, c2oMatrix = pipe.prepare_source_cvImage(sourceImage, realtime = True)

    if not ret:
        print(f"no face in source video...exit!")
        return None
    # Read video frame or looped
    dri_crop, out_crop, out_org, dri_motion_info = pipe.run(driving_frame, pipe.src_imgs[0], pipe.src_infos[0], first_frame=isFirstFrame)
    if out_crop is None:
        print("no face in driving image!")
        return None

    idx += 1

    out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)

    stitchedImage = StitchImage(sourceImage, out_crop, c2oMatrix)
    return stitchedImage;

async def handle_client(websocket, path):
    async for message in websocket:
        print("Received image data")

        if message == "RESET":
            print("Resetting the state...")
            global idx
            idx = 0
            continue
        elif message[0] == 0:
            global sourceImage;

            nparr = np.frombuffer(message[1:], np.uint8)
            sourceImage = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("Stored source image")
            continue
        elif message[0] == 1:
            # Process the image
            processed_image = process_image(message[1:])

            if processed_image is None:
                print("Failed to process image")
                processed_image = np.zeros((480, 640, 3), np.uint8)

            _, buffer = cv2.imencode('.jpg', processed_image)
            bytes = buffer.tobytes()

            await websocket.send(bytes)
            print("Sent processed grayscale image")

def StitchImage(originalImage, croppedImage, matrix):
    matrix = matrix.reshape(3, 3)
    
    originalHeight, originalWidth = originalImage.shape[:2]

    stitchedImage = np.zeros((originalHeight, originalWidth, 3), np.uint8)
    transformedCroppedImage = cv2.warpPerspective(croppedImage, matrix, (originalWidth, originalHeight))
    
    mask = cv2.cvtColor(transformedCroppedImage, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # apply erosion to mask to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(originalImage, originalImage, mask=mask_inv)
    foreground = cv2.bitwise_and(transformedCroppedImage, transformedCroppedImage, mask=mask)

    stitchedImage = cv2.add(background, foreground)

    # stitchedImage = cv2.rotate(stitchedImage, cv2.ROTATE_90_CLOCKWISE)

    return stitchedImage

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 9870, ping_timeout=None, ping_interval=None):
        print("Server started and waiting for connections...")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    args, unknown = parser.parse_known_args()

    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = False ## Default False

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal= False)

    asyncio.run(main())