import os
import random
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(BASE_DIR, "videos")
dataset_dir = os.path.join(BASE_DIR, "datasets")
temp_dir = os.path.join(BASE_DIR, "temp")
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"

width = 1800
height = 1000
RADIUS_OF_SEARCH_AREA = 11
should_draw_bbox = True
is_propagated = False
device = "cpu"

# if torch.cuda.is_available():
#     vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
#     if vram >= 1.5:
#         device = "cuda"

# if device == "cuda":
#     torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

frame_prompt_point_dict = dict()
current_index = 0
old_index = -1
prompts = dict()
video_segments = dict()
current_video_segment_mask = None
colors = {0: np.concatenate([np.array([255, 0, 0]), np.array([0.7])], axis=0)}
video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
img_formats = [".jpg", ".jpeg", ".png"]


def select_video():
    os.makedirs(videos_dir, exist_ok=True)

    videos = sorted(
        v
        for v in os.listdir(videos_dir)
        if os.path.splitext(v)[-1].lower() in video_formats
    )

    if len(videos) == 1:
        return os.path.join(videos_dir, videos[0])

    print("Videos:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video}")

    choice = input("Choose number: ")
    number = int(choice) - 1
    video_file = os.path.join(videos_dir, videos[number])

    return video_file


def select_frame_stride():
    return int(input("Frame stride (Enter = 1): ").strip() or "1")


def select_dataset_name():
    return input("Dataset name (Enter = 'object'): ").strip() or "object"


def transform_display():
    scale = min(width / frame_width, height / frame_height)
    new_w, new_h = int(frame_width * scale), int(frame_height * scale)
    offset_x = (width - new_w) // 2
    offset_y = (height - new_h) // 2
    return scale, new_w, new_h, offset_x, offset_y


def draw_points(img, points):
    for pos, label in points.items():
        point_color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(img, pos, RADIUS_OF_SEARCH_AREA, point_color, -1)


def draw_masks(img, mask_logits, obj_ids):
    for i, out_obj_id in enumerate(obj_ids):
        h, w = mask_logits[i].shape[-2:]
        mask = mask_logits[i].reshape(h, w, 1) * colors[out_obj_id].reshape(1, 1, -1)
        foreground_mask = mask[:, :, :3]
        alpha_channel = mask[:, :, 3]
        alpha_mask = alpha_channel[:, :, np.newaxis]
        img = img * (1 - alpha_mask) + foreground_mask * alpha_mask

    img = np.array(img, dtype=np.uint8)

    return img


def process_image(img):
    global frame_prompt_point_dict, current_index
    global prompts, inference_state
    global current_video_segment_mask

    promt_pos_lst = list()
    promt_labels_lst = list()
    obj_id = 0

    if len(frame_prompt_point_dict) > 0 and current_index in frame_prompt_point_dict:
        for pos, label in frame_prompt_point_dict[current_index].items():
            promt_pos_lst.append(list(pos))
            promt_labels_lst.append(label)

        points = np.array(promt_pos_lst, dtype=np.float32)
        labels = np.array(promt_labels_lst, np.int32)
        prompts[obj_id] = (points, labels)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=current_index,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        masks = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

        video_segments[current_index] = masks
        mask_logits = list(masks.values())
        current_video_segment_mask = mask_logits[0] if mask_logits else None
        img = draw_masks(img, mask_logits, out_obj_ids)
    else:
        current_video_segment_mask = video_segments.get(current_index, {}).get(0)
        if current_video_segment_mask is not None:
            img = draw_masks(img, [current_video_segment_mask], [0])

    return img


def refresh_image():
    global current_index, total
    global frame_names, frame_prompt_point_dict
    global should_draw_bbox
    global frame_width, frame_height
    global width, height

    img = cv2.imread(os.path.join(temp_dir, frame_names[current_index]))
    img = process_image(img)

    if current_index in frame_prompt_point_dict:
        points = frame_prompt_point_dict[current_index]
        draw_points(img, points)

    if should_draw_bbox and current_video_segment_mask is not None:
        bbox = find_box_in_mask(current_video_segment_mask)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    scale, new_w, new_h, offset_x, offset_y = transform_display()
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = cv2.resize(
        img, (new_w, new_h)
    )
    img = canvas

    num_hint = f"{current_index + 1}/{total}"
    cv2.putText(img, num_hint, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

    nav_hint = (
        "ANNOTATE: D/F=nav  LClick=pos  RClick=neg  R=reset  P=propagate"
        if not is_propagated
        else "REVIEW: D/F=nav  Click=fix  P=re-prop  E=export  H=bbox"
    )
    color = (0, 140, 255) if not is_propagated else (0, 200, 0)
    cv2.putText(img, nav_hint, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("auto_annotate_tracking", img)


def propagate_images():
    global is_propagated

    segments = dict()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    is_propagated = True

    return segments


def crop_image(image, bbox, crop_size=640):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if (x2 - x1) > crop_size or (y2 - y1) > crop_size or w < crop_size or h < crop_size:
        return None

    crop_x1 = int(np.clip((x1 + x2) // 2 - crop_size // 2, 0, w - crop_size))
    crop_y1 = int(np.clip((y1 + y2) // 2 - crop_size // 2, 0, h - crop_size))

    cropped = image[crop_y1 : crop_y1 + crop_size, crop_x1 : crop_x1 + crop_size]
    new_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
    return cropped, new_bbox


def mouse_callback(event, x, y, flags, param):
    global frame_width, frame_height
    global width, height
    global frame_prompt_point_dict
    global current_index

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        points = frame_prompt_point_dict.get(current_index, dict())

        scale, _, _, offset_x, offset_y = transform_display()
        x = max(0, min(int((x - offset_x) / scale), frame_width - 1))
        y = max(0, min(int((y - offset_y) / scale), frame_height - 1))

        founded_point = None
        for point in points.keys():
            x_t, y_t = point
            if (
                x >= x_t - RADIUS_OF_SEARCH_AREA - 1
                and x <= x_t + RADIUS_OF_SEARCH_AREA + 1
                and y >= y_t - RADIUS_OF_SEARCH_AREA - 1
                and y <= y_t + RADIUS_OF_SEARCH_AREA + 1
            ):
                founded_point = point
                break

        if event == cv2.EVENT_LBUTTONDOWN:
            point_value = 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_value = 0

        if founded_point is not None and points[founded_point] == point_value:
            del points[founded_point]
        else:
            points[(x, y)] = point_value

        if len(points) > 0:
            frame_prompt_point_dict[current_index] = points
        else:
            del frame_prompt_point_dict[current_index]

        refresh_image()


def find_box_in_mask(mask: np.array):
    coords = np.argwhere(mask[0])

    if coords.size == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return int(x_min), int(y_min), int(x_max), int(y_max)


def create_dataset(video_segments):
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)

    valid_frames = []
    for frame_idx in video_segments:
        mask = video_segments[frame_idx][0]
        bbox = find_box_in_mask(mask)

        if bbox is None:
            continue

        frame_filename = frame_names[frame_idx]
        image = cv2.imread(os.path.join(temp_dir, frame_filename))
        result = crop_image(image, bbox, crop_size=640)

        if result is None:
            continue

        cropped_image, cropped_bbox = result
        valid_frames.append((frame_filename, cropped_image, cropped_bbox))

    random.shuffle(valid_frames)
    total = len(valid_frames)

    val_count = max(1, round(total * 0.15))
    test_count = max(1, round(total * 0.15))
    train_count = max(0, total - val_count - test_count)
    train_end = train_count
    val_end = train_end + val_count
    test_end = val_end

    for split, frames in [
        ("train", valid_frames[:train_end]),
        ("val", valid_frames[train_end:val_end]),
        ("test", valid_frames[test_end:]),
    ]:
        for frame_filename, cropped_image, cropped_bbox in frames:
            label_filename = os.path.splitext(frame_filename)[0] + ".txt"

            x1, y1, x2, y2 = cropped_bbox
            crop_h, crop_w = cropped_image.shape[:2]

            x_center = ((x1 + x2) / 2) / crop_w
            y_center = ((y1 + y2) / 2) / crop_h
            w = (x2 - x1) / crop_w
            h = (y2 - y1) / crop_h

            with open(
                os.path.join(dataset_dir, "labels", split, label_filename), "w"
            ) as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

            cv2.imwrite(
                os.path.join(dataset_dir, "images", split, frame_filename),
                cropped_image,
            )

    yaml = (
        f"path: {dataset_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"  0: {dataset_name}\n"
    )

    with open(os.path.join(dataset_dir, "data.yaml"), "w") as f:
        f.write(yaml)

    print(f"Dataset created: train={train_count}, val={val_count}, test={test_count}")


video_file = select_video()
dataset_name = select_dataset_name()
frame_stride = select_frame_stride()

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

if frame_stride == 1:
    command = f'ffmpeg -i "{video_file}" -q:v 2 -start_number 0 "{temp_dir}/%05d.jpg"'
else:
    command = (
        f'ffmpeg -i "{video_file}"'
        f" -vf \"select='not(mod(n\\,{frame_stride}))'\" -vsync vfr"
        f' -q:v 2 -start_number 0 "{temp_dir}/%05d.jpg"'
    )
env = os.environ.copy()
env.pop("LD_LIBRARY_PATH", None)
subprocess.run(command, shell=True, env=env)

predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=temp_dir)

frame_names = [
    p for p in os.listdir(temp_dir) if os.path.splitext(p)[-1] in img_formats
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

frame_width, frame_height = Image.open(os.path.join(temp_dir, frame_names[0])).size

total = len(frame_names)
speed_change_frame = 1

cv2.namedWindow(
    "auto_annotate_tracking",
    flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
)
cv2.setMouseCallback("auto_annotate_tracking", mouse_callback)

while True:
    if old_index != current_index:
        old_index = current_index
        refresh_image()

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("d"):
        step = min(speed_change_frame, current_index)
        current_index -= step
        speed_change_frame = 10
    elif key == ord("f") and current_index < total - 1:
        step = min(speed_change_frame, total - current_index - 1)
        current_index += step
        speed_change_frame = 10
    elif key == ord("r"):
        del frame_prompt_point_dict[current_index]
        refresh_image()
    elif key == ord("h"):
        should_draw_bbox = not should_draw_bbox
        refresh_image()
    elif key == ord("p"):
        video_segments = propagate_images()
        refresh_image()
    elif key == ord("e"):
        create_dataset(video_segments)
    elif key == -1:
        speed_change_frame = 1


cv2.destroyAllWindows()

print("Program finished")
