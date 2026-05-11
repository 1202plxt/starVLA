import dataclasses
import json
import logging
import math
import os
import pathlib
import time
import base64
from io import BytesIO

import imageio
import numpy as np
import tqdm
import tyro
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)

# ============================================================================
# ========== [新增] 3D 到 2D 坐标投影及物体匹配 ==============================
def get_pixel_coords(sim, body_name, camera_name="agentview", width=256, height=256):
    if body_name is None: return [width // 2, height // 2]
    try:
        body_id = sim.model.body_name2id(body_name)
        pos = sim.data.body_xpos[body_id]
        cam_id = sim.model.camera_name2id(camera_name)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)

        pos_cam = cam_mat.T @ (pos - cam_pos)
        fovy = sim.model.cam_fovy[cam_id]
        f = 0.5 * height / math.tan(fovy * math.pi / 360)
        
        u = int((pos_cam[0] / -pos_cam[2]) * f + width / 2.0)
        v = int((-pos_cam[1] / -pos_cam[2]) * f + height / 2.0)

        u_flipped = width - 1 - max(0, min(u, width - 1))
        v_flipped = height - 1 - max(0, min(v, height - 1))
        return [u_flipped, v_flipped]
    except Exception as e:
        return [width // 2, height // 2]

def _get_target_object_name(task_description, body_names):
    desc = task_description.lower()
    target_keyword = None
    if "bowl" in desc: target_keyword = "bowl"
    elif "plate" in desc: target_keyword = "plate"
    elif "wine" in desc or "bottle" in desc: target_keyword = "wine_bottle"
    elif "frypan" in desc or "stove" in desc: target_keyword = "frypan"
    elif "cheese" in desc: target_keyword = "cream_cheese"
    
    if target_keyword:
        for name in body_names:
            if target_keyword in name.lower() and "body" in name.lower(): return name
    return None
# ============================================================================

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224, 224]
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "experiments/libero/logs"
    seed: int = 7
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test"

def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial": max_steps = 220
    elif args.task_suite_name == "libero_object": max_steps = 280
    elif args.task_suite_name == "libero_goal": max_steps = 300
    elif args.task_suite_name == "libero_10": max_steps = 520
    elif args.task_suite_name == "libero_90": max_steps = 400
    else: raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        policy_ckpt_path=args.pretrained_path,
        host=args.host, port=args.port, image_size=args.resize_size,
    )

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            client_model.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            full_actions = []
            step = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                state = np.concatenate((
                    obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"],
                ))

                observation = {
                    "observation.primary": np.expand_dims(img, axis=0),
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # 发送坐标提示
                available_body_names = env.env.sim.model.body_names
                target_object_name = _get_target_object_name(task_description, available_body_names)
                prompt_uv = get_pixel_coords(
                    sim=env.env.sim, body_name=target_object_name, 
                    camera_name="agentview", width=LIBERO_ENV_RESOLUTION, height=LIBERO_ENV_RESOLUTION
                )

                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                    "sam2_prompt": prompt_uv 
                }

                response = client_model.step(example=example_dict, step=step)
                raw_action = response["raw_action"]

                # ============================================================================
                # ========== [修改] 提取服务端回传的高亮图并存入视频 ==========
                vis_img_saved = False
                target_dict_b64 = raw_action if "sam2_vis_b64" in raw_action else response
                
                if "sam2_vis_b64" in target_dict_b64:
                    try:
                        img_data = base64.b64decode(target_dict_b64["sam2_vis_b64"])
                        vis_img = np.array(Image.open(BytesIO(img_data)))
                        replay_images.append(vis_img) 
                        vis_img_saved = True
                    except Exception as e:
                        logging.warning(f"Failed to decode returned image: {e}")
                
                if not vis_img_saved:
                    replay_images.append(img) # 降级：如果没有收到高亮图，则保存素颜图
                # ============================================================================

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    raise ValueError("Invalid action sizes")
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                obs, reward, done, info = env.step(delta_action.tolist())
                
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images], fps=10,
            )
            logging.info(f"Success: {done}")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0): return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False): return
    debugpy.listen(("0.0.0.0", 10092))
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", False): start_debugpy_once()
    tyro.cli(eval_libero)
