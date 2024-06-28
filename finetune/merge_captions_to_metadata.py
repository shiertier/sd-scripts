import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import library.train_util as train_util
import os
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def main(args):
    assert not args.recursive or (
        args.recursive and args.full_path
    ), "recursive(递归) 需要 full_path"

    train_data_dir_path = Path(args.train_data_dir)
    image_paths: List[Path] = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
    logger.info(f"找到 {len(image_paths)} 张图片.")

    if args.in_json is None and Path(args.out_json).is_file():
        args.in_json = args.out_json

    if args.in_json is not None:
        logger.info(f"加载已存在 metadata: {args.in_json}")
        metadata = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
        logger.warning(" 请注意，您所做的更改将覆盖现有图片的描述。")
    else:
        logger.info("将会创建新的元数据")
        metadata = {}

    logger.info("merge caption texts to metadata json.")
    for image_path in tqdm(image_paths):
        caption_path = image_path.with_suffix(args.caption_extension)
        caption = caption_path.read_text(encoding="utf-8").strip()

        if not os.path.exists(caption_path):
            caption_path = os.path.join(image_path, args.caption_extension)

        image_key = str(image_path) if args.full_path else image_path.stem
        if image_key not in metadata:
            metadata[image_key] = {}

        metadata[image_key]["caption"] = caption
        if args.debug:
            logger.info(f"{image_key} {caption}")

    # 写入metadata然后结束
    logger.info(f"写入 元数据: {args.out_json}")
    Path(args.out_json).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="训练图像的目录")
    parser.add_argument("out_json", type=str, help="要输出的元数据文件")
    parser.add_argument(
        "--in_json",
        type=str,
        help="要输入的元数据文件（如果省略，并且存在out_json，则读取现有的out_json）",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="字幕文件的扩展名（为了向后兼容）",
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="字幕文件的扩展名"
    )
    parser.add_argument(
        "--full_path",
        action="store_true",
        help="在元数据中使用完整路径作为图像键（支持多个目录）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归地在train_data_dir的所有子文件夹中查找训练标签",
    )
    parser.add_argument("--debug", action="store_true", help="debug 模式")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # 恢复拼写错误的选项
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)
