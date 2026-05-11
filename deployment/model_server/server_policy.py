# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image

# ========== [新增] 引入 SAM 2 核心库 ==========
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
# ==============================================

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework


# ============================================================================
# ========== [修改后] SAM 2 视觉提示拦截器 (带深度搜索 & 强制落盘) ==========
# ============================================================================
class SAM2PolicyWrapper:
    """
    代理类：用于拦截客户端请求，应用 SAM 2 视觉提示，
    支持在任意复杂度的嵌套 JSON 中精准定位 prompt。
    """
    def __init__(self, vla_model):
        self.vla_model = vla_model
        logging.info("Initializing SAM 2 Visual Prompt Module...")
        
        sam2_checkpoint = os.path.expanduser("~/autodl-tmp/sam2/checkpoints/sam2_hiera_small.pt")
        model_cfg = "sam2_hiera_s.yaml" 
        
        self.sam2_predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device="cuda"))
        self.saved_instructions = set() # 记忆集合，防重复存图
        logging.info("SAM 2 Initialization Complete. Wrapper deployed.")

    def step(self, *args, **kwargs):
        return self._process_and_forward('step', *args, **kwargs)

    def predict_action(self, *args, **kwargs):
        return self._process_and_forward('predict_action', *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.vla_model, name)

    def _process_and_forward(self, method, *args, **kwargs):
        
        # 1. 递归寻找包含 "sam2_prompt" 的核心字典 (打破层层包裹)
        def find_prompt_dict(data):
            if isinstance(data, dict):
                if "sam2_prompt" in data:
                    return data
                for val in data.values():
                    res = find_prompt_dict(val)
                    if res: return res
            elif isinstance(data, list) or isinstance(data, tuple):
                for item in data:
                    res = find_prompt_dict(item)
                    if res: return res
            return None

        target_dict = find_prompt_dict(kwargs)
        highlighted_img_np = None  
        
        # 2. 如果找到了核心字典，立刻开始劫持
        if target_dict is not None:
            logging.info("🎯 [精确制导] 成功从层层包裹中挖出了 sam2_prompt!")
            
            prompt_uv = target_dict.pop("sam2_prompt") # 取出坐标
            images = target_dict.get("image", [])
            text_key = "lang" if "lang" in target_dict else "instruction"
            instruction = target_dict.get(text_key, "")
            
            if prompt_uv is not None and len(images) > 0:
                try:
                    import os
                    from PIL import Image
                    import numpy as np
                    
                    img_obj = images[0]
                    img = np.array(img_obj) if isinstance(img_obj, Image.Image) else np.array(img_obj).copy()
                        
                    # SAM 2 推理
                    self.sam2_predictor.set_image(img)
                    masks, _, _ = self.sam2_predictor.predict(
                        point_coords=np.array([prompt_uv]),
                        point_labels=np.array([1]), 
                        multimask_output=False
                    )
                    best_mask = masks[0]
                    
                    # 应用绿色高亮掩码
                    img[best_mask > 0] = [0, 255, 0] 
                    highlighted_img_np = img  
                    
                    # 强制落盘保存单张高亮调试图
                    if instruction not in self.saved_instructions:
                        save_dir = "sam2_debug_images"
                        os.makedirs(save_dir, exist_ok=True)
                        safe_name = "".join([c if c.isalnum() else "_" for c in instruction])[:50]
                        save_path = os.path.join(save_dir, f"{safe_name}.jpg")
                        
                        Image.fromarray(highlighted_img_np).save(save_path, quality=95)
                        logging.info(f"📸 [强制落盘] 成功保存高亮验证图至: {save_path}")
                        self.saved_instructions.add(instruction)
                    
                    # 把处理好的图和新指令塞回原来的字典里
                    target_dict["image"][0] = Image.fromarray(img) if isinstance(img_obj, Image.Image) else img
                    target_dict[text_key] = instruction + " Notice the target object highlighted in green."
                    
                except Exception as e:
                    logging.error(f"[SAM 2 Error] Processing failed: {e}")
                    
        # 3. 让原始模型推理
        func = getattr(self.vla_model, method)
        action_result = func(*args, **kwargs)
        
        # 4. 尝试 Base64 回传
        if highlighted_img_np is not None and isinstance(action_result, dict):
            try:
                import base64
                from io import BytesIO
                from PIL import Image
                buffered = BytesIO()
                Image.fromarray(highlighted_img_np).save(buffered, format="JPEG", quality=85)
                action_result["sam2_vis_b64"] = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                pass

        return action_result
# ============================================================================

def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained(  # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.use_bf16:  # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    # ========================================================================
    # ========== [新增] 将 SAM 2 包装器应用于 VLA 模型 ==========
    vla = SAM2PolicyWrapper(vla)
    # ========================================================================

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
