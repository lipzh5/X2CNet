import os
import os.path as osp
import tyro
from liveportrait_configs.argument_config import ArgumentConfig
from liveportrait_configs.inference_config import InferenceConfig
from liveportrait_configs.crop_config import CropConfig
from x2cnet_pipeline import X2CNetPipeline
import time
from pathlib import Path
CUR_DIR = Path(__file__).resolve().parent


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    args.source = 'assets/ameca_neutral.jpg' # osp.join(CUR_DIR, 'assets/ameca_neutral.jpg') 
    # args.driving_image_folder = 'assets/driving/example'
    # args.driving = '/home/penny/pycharmprojects/liveportrait/assets/examples/driving2/12FPS/penny2.mp4'
    args.output_dir = '/home/penny/pycharmprojects/assets/animations2'
    args.flag_do_crop = False
    args.export_video = True

    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)
    # inference_cfg.checkpoint_G = '/home/penny/pycharmprojects/humanoidexpgen/checkpoints/train_spade/8_net_G.pth'
    # # inference_cfg.checkpoint_W = '/home/penny/pycharmprojects/humanoidexpgen/checkpoints/train_warp_spade/4_net_W.pth'
    # # inference_cfg.checkpoint_G = '/home/penny/pycharmprojects/humanoidexpgen/checkpoints/train_warp_spade/4_net_G.pth'
    # inference_cfg.checkpoint_X = '/home/penny/pycharmprojects/humanoidexpgen/saved_models/trial_000_bs128_ep100_lr0.001_lossfn2_enc-resnet_pt1_raug0_norm-1_phys0_size224/x2control.pth'
    # inference_cfg.mapping_net_cfg = 'conf/config.yaml'

    pipeline = X2CNetPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    ctrl_values = pipeline.execute(args) 
    print(f'len all ctrls: {len(ctrl_values)}, {ctrl_values[0].shape}') 
    


if __name__ == "__main__":
    main()