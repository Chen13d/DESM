from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile


DEFAULT_INPUT = Path(
    r"path\to\input.tif"
)
DEFAULT_INFERENCE = Path(
    r"path\to\inference.tif"
)
DEFAULT_OUTPUT_NAME = "Input_Inference_concat.tif"
POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Color must be R,G,B with values from 0 to 255.")
    return parts[0], parts[1], parts[2]


def load_font(font_size: int, font_path: Path | None = None) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), font_size)

    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_bbox_and_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox, (bbox[2] - bbox[0], bbox[3] - bbox[1])


def draw_text_by_ink_box(
    draw: ImageDraw.ImageDraw,
    ink_x: int,
    ink_y: int,
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
) -> None:
    bbox, _ = text_bbox_and_size(draw, text, font)
    draw.text((ink_x - bbox[0], ink_y - bbox[1]), text, fill=color, font=font)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    _, size = text_bbox_and_size(draw, text, font)
    return size


def resize_spatial(image: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return image

    frame_shape = image.shape[-3:]
    frames = image.reshape((-1, *frame_shape))
    resized_frames = []
    for frame in frames:
        pil_frame = Image.fromarray(frame)
        new_size = (pil_frame.width * scale, pil_frame.height * scale)
        resized_frames.append(np.asarray(pil_frame.resize(new_size, Image.Resampling.BICUBIC)))

    new_frame_shape = resized_frames[0].shape
    return np.stack(resized_frames, axis=0).reshape((*image.shape[:-3], *new_frame_shape))


def scaled(value: int, scale: int, minimum: int = 1) -> int:
    return max(minimum, int(round(value * scale)))


def scaled_font(
    font_size: int, scale: int, font_path: Path | None = None
) -> ImageFont.ImageFont:
    return load_font(scaled(font_size, scale), font_path)


def scaled_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color


def anchor_xy(
    position: str,
    canvas_width: int,
    canvas_height: int,
    object_width: int,
    object_height: int,
    margin_x: int,
    margin_y: int,
) -> tuple[int, int]:
    if "left" in position:
        x = margin_x
    else:
        x = canvas_width - margin_x - object_width

    if "top" in position:
        y = margin_y
    else:
        y = canvas_height - margin_y - object_height
    return max(0, x), max(0, y)


def horizontal_axis(array: np.ndarray) -> int:
    """Return the width axis for grayscale stacks or RGB/RGBA TIFF arrays."""
    if array.ndim >= 3 and array.shape[-1] in (3, 4):
        return array.ndim - 2
    return array.ndim - 1


def concat_axis_for_direction(array: np.ndarray, direction: str) -> int:
    width_axis = horizontal_axis(array)
    if direction == "horizontal":
        return width_axis

    height_axis = width_axis - 1
    if height_axis < 0:
        raise ValueError(f"Cannot infer vertical axis for shape={array.shape}")
    return height_axis


def match_channels(input_img: np.ndarray, inference_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert grayscale/RGB pairs to the same channel layout when possible."""
    input_is_color = input_img.ndim >= 3 and input_img.shape[-1] in (3, 4)
    inference_is_color = inference_img.ndim >= 3 and inference_img.shape[-1] in (3, 4)

    if input_is_color == inference_is_color:
        return input_img, inference_img

    if inference_is_color and input_img.shape == inference_img.shape[:-1]:
        channels = inference_img.shape[-1]
        input_img = np.repeat(input_img[..., None], channels, axis=-1)
        return input_img, inference_img

    if input_is_color and inference_img.shape == input_img.shape[:-1]:
        channels = input_img.shape[-1]
        inference_img = np.repeat(inference_img[..., None], channels, axis=-1)
        return input_img, inference_img

    return input_img, inference_img


def ensure_rgb_for_annotation(image: np.ndarray) -> np.ndarray:
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        return image
    return np.repeat(image[..., None], 3, axis=-1)


def make_same_channels(images: list[np.ndarray]) -> list[np.ndarray]:
    has_color = any(image.ndim >= 3 and image.shape[-1] in (3, 4) for image in images)
    if not has_color:
        return images

    color_channels = next(image.shape[-1] for image in images if image.ndim >= 3 and image.shape[-1] in (3, 4))
    matched = []
    for image in images:
        if image.ndim >= 3 and image.shape[-1] in (3, 4):
            if image.shape[-1] != color_channels:
                raise ValueError(f"Mixed color channel counts are not supported: shape={image.shape}")
            matched.append(image)
        else:
            matched.append(np.repeat(image[..., None], color_channels, axis=-1))
    return matched


def draw_scale_bar(
    frame: np.ndarray,
    *,
    bar_px: int,
    bar_height: int,
    label: str,
    position: str,
    margin_x: int,
    margin_y: int,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    label_gap: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    label_width, label_height = text_size(draw, label, font) if label else (0, 0)
    object_width = max(bar_px, label_width)
    object_height = bar_height + (label_gap + label_height if label else 0)
    x, y = anchor_xy(position, image.width, image.height, object_width, object_height, margin_x, margin_y)
    bar_x = x + (object_width - bar_px) // 2

    if "bottom" in position and label:
        label_x = x + (object_width - label_width) // 2
        label_y = y
        bar_y = y + label_height + label_gap
    else:
        bar_y = y
        label_x = x + (object_width - label_width) // 2
        label_y = y + bar_height + label_gap

    draw.rectangle([bar_x, bar_y, bar_x + bar_px - 1, bar_y + bar_height - 1], fill=color)
    if label:
        draw_text_by_ink_box(draw, label_x, label_y, label, font, color)
    return np.asarray(image)


def draw_timestamp(
    frame: np.ndarray,
    *,
    text: str,
    position: str,
    margin_x: int,
    margin_y: int,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width, height = text_size(draw, text, font)
    x, y = anchor_xy(position, image.width, image.height, width, height, margin_x, margin_y)
    draw_text_by_ink_box(draw, x, y, text, font, color)
    return np.asarray(image)


def add_annotations(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.no_scale_bar and args.no_timestamp:
        return image

    annotated = ensure_rgb_for_annotation(image)
    annotated = resize_spatial(annotated, args.annotation_upscale)
    frame_shape = annotated.shape[-3:]
    if frame_shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA data after conversion, got shape={annotated.shape}")

    frames = annotated.reshape((-1, *frame_shape))
    scale = args.annotation_upscale
    scale_font = scaled_font(args.scale_bar_font_size, scale, args.font_path)
    timestamp_font = scaled_font(args.timestamp_font_size, scale, args.font_path)

    for frame_index in range(frames.shape[0]):
        frame = frames[frame_index]
        if not args.no_scale_bar:
            frame = draw_scale_bar(
                frame,
                bar_px=scaled(args.scale_bar_px, scale),
                bar_height=scaled(args.scale_bar_height, scale),
                label=args.scale_bar_label,
                position=args.scale_bar_position,
                margin_x=scaled(args.scale_bar_margin_x, scale, 0),
                margin_y=scaled(args.scale_bar_margin_y, scale, 0),
                color=scaled_color(args.scale_bar_color),
                font=scale_font,
                label_gap=scaled(args.scale_bar_label_gap, scale),
            )

        if not args.no_timestamp:
            time_value = args.time_start + frame_index * args.time_interval
            timestamp = args.timestamp_text.format(frame=frame_index, time=time_value)
            frame = draw_timestamp(
                frame,
                text=timestamp,
                position=args.timestamp_position,
                margin_x=scaled(args.timestamp_margin_x, scale, 0),
                margin_y=scaled(args.timestamp_margin_y, scale, 0),
                color=scaled_color(args.timestamp_color),
                font=timestamp_font,
            )

        frames[frame_index] = frame

    return annotated


def concat_tifs(
    input_path: Path,
    inference_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    middle_path: Path | None = None,
) -> None:
    paths = [input_path]
    if middle_path is not None:
        paths.append(middle_path)
    paths.append(inference_path)

    images = [tifffile.imread(path) for path in paths]
    dtype = images[0].dtype
    images = [image.astype(dtype, copy=False) if image.dtype != dtype else image for image in images]
    images = make_same_channels(images)

    concat_axis = concat_axis_for_direction(images[0], args.concat_direction)
    check_shape = list(images[0].shape)
    check_shape[concat_axis] = -1
    for path, image in zip(paths[1:], images[1:]):
        image_concat_axis = concat_axis_for_direction(image, args.concat_direction)
        image_check_shape = list(image.shape)
        image_check_shape[image_concat_axis] = -1
        if concat_axis != image_concat_axis or check_shape != image_check_shape:
            shapes = ", ".join(f"{path.name}={image.shape}" for path, image in zip(paths, images))
            raise ValueError(
                f"TIFF shapes must match except concat axis for {args.concat_direction}: {shapes}"
            )

    merged = np.concatenate(images, axis=concat_axis)
    merged = add_annotations(merged, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_path, merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate an input TIFF and inference TIFF horizontally."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input.tif path")
    parser.add_argument(
        "--inference", type=Path, default=DEFAULT_INFERENCE, help="Inference.tif path"
    )
    parser.add_argument(
        "--middle",
        type=Path,
        default=None,
        help="Optional middle TIFF path. Useful for Input | FLIM | Inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output tif path. Defaults to Input.tif folder/Input_Inference_concat.tif",
    )
    parser.add_argument("--font-path", type=Path, default=None, help="Optional .ttf font path")
    parser.add_argument(
        "--annotation-upscale",
        type=int,
        default=4,
        help="Upsample each frame before drawing annotations. Scale bar pixels are in original-image pixels.",
    )

    parser.add_argument("--no-scale-bar", action="store_true", help="Disable scale bar")
    parser.add_argument("--scale-bar-px", type=int, default=20, help="Scale bar width in pixels")
    parser.add_argument("--scale-bar-height", type=int, default=3, help="Scale bar height in pixels")
    parser.add_argument("--scale-bar-label", default="10 um", help="Scale bar label text")
    parser.add_argument("--scale-bar-position", choices=POSITIONS, default="bottom-left")
    parser.add_argument("--scale-bar-margin-x", type=int, default=6)
    parser.add_argument("--scale-bar-margin-y", type=int, default=6)
    parser.add_argument("--scale-bar-label-gap", type=int, default=6)
    parser.add_argument("--scale-bar-font-size", type=int, default=10)
    parser.add_argument("--scale-bar-color", type=parse_rgb, default=(255, 255, 255))

    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp")
    parser.add_argument("--time-interval", type=float, default=1.0, help="Time interval per frame")
    parser.add_argument("--time-start", type=float, default=0.0, help="Timestamp start time")
    parser.add_argument(
        "--timestamp-text",
        default="{time:.1f} s",
        help="Timestamp format string. Available fields: {frame}, {time}",
    )
    parser.add_argument("--timestamp-position", choices=POSITIONS, default="top-left")
    parser.add_argument("--timestamp-margin-x", type=int, default=6)
    parser.add_argument("--timestamp-margin-y", type=int, default=6)
    parser.add_argument("--timestamp-font-size", type=int, default=10)
    parser.add_argument("--timestamp-color", type=parse_rgb, default=(255, 255, 255))

    parser.add_argument(
        "--concat-direction",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="horizontal: left-right; vertical: top-bottom",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_name(DEFAULT_OUTPUT_NAME)
    concat_tifs(args.input, args.inference, output, args, middle_path=args.middle)
    print(f"Saved: {output}")


if __name__ == "__main__":
    # Mito Lyso motion 1
    if 1:
        input_path = Path(
            r"path\to\mito_lyso_tile1_input.tif"
        )
        inference_path = Path(
            r"path\to\mito_lyso_tile1_inference.tif"
        )
        output_path = input_path.with_name("Input_Inference_concat.tif")

        args = parse_args()
        args.input = input_path
        args.inference = inference_path
        args.output = output_path

        # Scale bar
        args.no_scale_bar = False
        args.scale_bar_px = 25          # scale bar 在图上的像素长度
        args.scale_bar_height = 3       # scale bar 粗细
        args.scale_bar_label = f"{args.scale_bar_px * 20} nm"  # scale bar 代表的真实长度
        args.scale_bar_position = "bottom-left"
        args.scale_bar_margin_x = 6
        args.scale_bar_margin_y = 6

        # Time stamp
        args.no_timestamp = False
        args.time_start = 30
        args.time_interval = 30
        args.timestamp_text = "{time:.0f} s"
        args.timestamp_position = "top-left"
        args.timestamp_margin_x = 6
        args.timestamp_margin_y = 6

        concat_tifs(args.input, args.inference, args.output, args)
        print(f"Saved: {args.output}")

    # Mito Lyso motion 2
    if 1:
        input_path = Path(
            r"path\to\mito_lyso_tile2_input.tif"
        )
        inference_path = Path(
            r"path\to\mito_lyso_tile2_inference.tif"
        )
        output_path = input_path.with_name("Input_Inference_concat.tif")

        args = parse_args()
        args.input = input_path
        args.inference = inference_path
        args.output = output_path

        # Scale bar
        args.no_scale_bar = False
        args.scale_bar_px = 25          # scale bar 在图上的像素长度
        args.scale_bar_height = 3       # scale bar 粗细
        args.scale_bar_label = f"{args.scale_bar_px * 20} nm"  # scale bar 代表的真实长度
        args.scale_bar_position = "bottom-left"
        args.scale_bar_margin_x = 6
        args.scale_bar_margin_y = 6

        # Time stamp
        args.no_timestamp = False
        args.time_start = 30
        args.time_interval = 30
        args.timestamp_text = "{time:.0f} s"
        args.timestamp_position = "top-left"
        args.timestamp_margin_x = 6
        args.timestamp_margin_y = 6

        args.concat_direction = "vertical"

        concat_tifs(args.input, args.inference, args.output, args)
        print(f"Saved: {args.output}")
    
    # Micro Lyso motion 1
    if 1:
        input_path = Path(
            r"path\to\micro_lyso_tile1_input.tif"
        )
        inference_path = Path(
            r"path\to\micro_lyso_tile1_inference.tif"
        )
        output_path = input_path.with_name("Input_Inference_concat.tif")

        args = parse_args()
        args.input = input_path
        args.inference = inference_path
        args.output = output_path

        # Scale bar
        args.no_scale_bar = False
        args.scale_bar_px = 25          # scale bar 在图上的像素长度
        args.scale_bar_height = 3       # scale bar 粗细
        args.scale_bar_label = f"{args.scale_bar_px * 20} nm"  # scale bar 代表的真实长度
        args.scale_bar_position = "bottom-left"
        args.scale_bar_margin_x = 6
        args.scale_bar_margin_y = 6

        # Time stamp
        args.no_timestamp = False
        args.time_start = 3
        args.time_interval = 3
        args.timestamp_text = "{time:.0f} s"
        args.timestamp_position = "top-left"
        args.timestamp_margin_x = 6
        args.timestamp_margin_y = 6

        concat_tifs(args.input, args.inference, args.output, args)
        print(f"Saved: {args.output}")

    # Micro Lyso motion 2
    if 1:
        input_path = Path(
            r"path\to\micro_lyso_tile2_input.tif"
        )
        inference_path = Path(
            r"path\to\micro_lyso_tile2_inference.tif"
        )
        output_path = input_path.with_name("Input_Inference_concat.tif")

        args = parse_args()
        args.input = input_path
        args.inference = inference_path
        args.output = output_path

        # Scale bar
        args.no_scale_bar = False
        args.scale_bar_px = 20          # scale bar 在图上的像素长度
        args.scale_bar_height = 3       # scale bar 粗细
        args.scale_bar_label = f"{args.scale_bar_px * 20} nm"  # scale bar 代表的真实长度
        args.scale_bar_position = "bottom-left"
        args.scale_bar_margin_x = 0
        args.scale_bar_margin_y = 6
        args.scale_bar_font_size = 7

        # Time stamp
        args.no_timestamp = False
        args.time_start = 3
        args.time_interval = 3
        args.timestamp_text = "{time:.0f} s"
        args.timestamp_position = "top-left"
        args.timestamp_margin_x = 6
        args.timestamp_margin_y = 6
        args.timestamp_font_size = 7


        concat_tifs(args.input, args.inference, args.output, args)
        print(f"Saved: {args.output}")

    # IL Mito Lyso motion 1
    if 1:
        input_path = Path(
            r"path\to\desm_il_mito_lyso_input.tif"
        )
        inference_path = Path(
            r"path\to\desm_il_mito_lyso_inference.tif"
        )
        flim_path = Path(
            r"path\to\desm_il_mito_lyso_flim.tif"
        )
        output_path = input_path.with_name("Input_FLIM_Inference_concat.tif")

        args = parse_args()
        args.input = input_path
        args.middle = flim_path
        args.inference = inference_path
        args.output = output_path

        # Scale bar
        args.no_scale_bar = False
        args.scale_bar_px = 25          # scale bar 在图上的像素长度
        args.scale_bar_height = 3       # scale bar 粗细
        args.scale_bar_label = f"{args.scale_bar_px * 20} nm"  # scale bar 代表的真实长度
        args.scale_bar_position = "bottom-left"
        args.scale_bar_margin_x = 6
        args.scale_bar_margin_y = 6

        # Time stamp
        args.no_timestamp = False
        args.time_start = 1
        args.time_interval = 1
        args.timestamp_text = "{time:.0f} min"
        args.timestamp_position = "top-left"
        args.timestamp_margin_x = 6
        args.timestamp_margin_y = 6

        concat_tifs(args.input, args.inference, args.output, args, middle_path=args.middle)
        print(f"Saved: {args.output}")
