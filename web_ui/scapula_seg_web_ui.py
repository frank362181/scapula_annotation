
"""
    --Web UI
    intelligent annotation system for shoulder blade on Chest X-ray Images of Pneumoconiosis
"""

import cv2
import gradio as gr

from typing import Optional

import numpy as np
from engine.scapula_seg_engine import ScapulaSegEngine


class ScapulaSegWebUI:
    def __init__(self,config_path: str, checkpoint_path: str,device: str="cuda:0",
                 default_opacity: float = 0.4,engine: Optional[ScapulaSegEngine]=None):

        self._default_opacity = default_opacity
        if engine is None:
            self._engine = ScapulaSegEngine(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                device=device,
                over_opacity=default_opacity,
            )
        else:
            self._engine = engine
        
        # build the UI
        self._app = self._build_interface()
    
    def launch(self,**kwargs):
        kwargs.setdefault("server_name","0.0.0.0")
        kwargs.setdefault("server_port",7860)
        kwargs.setdefault("share", False)

        self._app.launch(**kwargs)



    """================== private functions & methods =================="""
    def _build_interface(self) -> gr.Blocks:
        with gr.Blocks() as app:
            gr.Markdown("## 尘肺胸片肩胛骨智能标注系统")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="上传胸片图像",type="numpy")
                    opacity_slider = gr.Slider(minimum=0.1,maximum=0.9,value=self._default_opacity,step=0.05,label="叠加图透明度")
                    submit_btn = gr.Button("开始分析",variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="标注结果",type="numpy")
                    output_report = gr.Markdown(label="分析报告")
            
            submit_btn.click(
                fn=self._analyze_image,
                inputs=[input_image,opacity_slider],
                outputs=[output_image,output_report]
            )

        return app
    
    def _analyze_image(self,image:np.ndarray,opacity: float) -> tuple:
        if image is None:
            return None,"请上传有效的胸片图像！"
        
        image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        self._engine._render.set_opacity(opacity)

        result = self._engine.run(image_bgr,return_overlay=True)

        # 2 ---- convert to RGB for display
        overlay_rgb = cv2.cvtColor(result["overlay"],cv2.COLOR_BGR2RGB)

        # 3 ---- generate report
        total_pixels = result["original"].shape[0] * result["original"].shape[1]
        ratio = result["area_px"] / total_pixels * 100
        report = f"""
        ### 分析报告
        - 肩胛骨像素面积: {result["area_px"]} px
        - 肩胛骨占比: {ratio:.2f} %
        - 推理状态: 完成
        """

        return overlay_rgb,report