
import os

from web_ui.scapula_seg_web_ui import ScapulaSegWebUI


def main():
    web_ui = ScapulaSegWebUI(
        config_path='configs/scapula_seg_swin_tiny.py',
        checkpoint_path='checkpoints/best_model.pth',
        device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
        default_opacity=0.4
    )

    web_ui.launch()

if __name__ == '__main__':
    main()

