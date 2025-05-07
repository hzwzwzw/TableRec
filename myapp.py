import time

import cv2
import gradio as gr
from lineless_table_rec import LinelessTableRecognition
from wired_table_rec import WiredTableRecognition
from wired_table_rec.main import WiredTableInput

from utils import plot_rec_box, LoadImage, format_html, box_4_2_poly_to_box_4_1

import mywired

img_loader = LoadImage()
# table_rec_path = "models/tsr/ch_ppstructure_mobile_v2_SLANet.onnx"
det_model_dir = {
    "mobile_det": "models/ocr/ch_PP-OCRv4_det_server_infer.onnx",
    # "mobile_det": "modeltest/cv_resnet18_ocr-detection-db-line-level_damo/model.onnx",
}

rec_model_dir = {
    "mobile_rec": "models/ocr/ch_PP-OCRv4_rec_server_infer.onnx",
    # "mobile_rec": "modeltest/cv_convnextTiny_ocr-recognition-general_damo/model.onnx",
}
table_engine_list = [
    "auto",
    "RapidTable(SLANet)",
    "RapidTable(SLANet-plus)",
    "RapidTable(unitable)",
    "wired_table_v2",
    "wired_table_v1",
    "lineless_table"
]

# 示例图片路径
example_images = [
    "images/10.jpg",
    "images/JC.png"
]
config = WiredTableInput(model_path=None)
table_engine = mywired.mywired(config)

# from preproc.scan import DocScanner
# scanner = DocScanner()
# from paddlex import create_model
# uvdoc = create_model(model_name="UVDoc")

from rapid_undistorted.inference import InferenceEngine
scanengine = InferenceEngine()

def process_image(img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotated_fix, col_threshold, row_threshold, remove_line, preproc, visual=True):
    img = img_loader(img_input)
    if preproc:
        img, _ = scanengine(img, ["unwrap", "unshadow", ("unblur", "OpenCvBilateral")])
        # img = ocrtest.proc(img)
        pass
    # output = uvdoc.predict(img)
    # for res in output:
    #     img = res['doctr_img']
    start = time.time()
    talbe_type = "wired_table_v2"
    det_cost, cls_cost, rec_cost = 0, 0, 0
    assert isinstance(table_engine, (WiredTableRecognition, LinelessTableRecognition))
    html, table_rec_elapse, polygons, logic_points, ocr_res = table_engine(img, ocr_result=None,
                                                                                   enhance_box_line=small_box_cut_enhance,
                                                                                   rotated_fix=rotated_fix,
                                                                                   col_threshold=col_threshold,
                                                                                   row_threshold=row_threshold)
    sum_elapse = time.time() - start
    all_elapse = f"- table_type: {talbe_type}\n table all cost: {sum_elapse:.5f}\n - table rec cost: {table_rec_elapse[0]:.5f}\n - ocr cost: {det_cost + cls_cost + rec_cost:.5f}"

    if visual:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        table_boxes_img = plot_rec_box(img.copy(), polygons)
        ocr_boxes_img = plot_rec_box(img.copy(), ocr_res)
        complete_html = format_html(html)

    return complete_html, table_boxes_img, ocr_boxes_img, table_rec_elapse, all_elapse


def main():

    with gr.Blocks(css="""
        .scrollable-container {
            overflow-x: auto;
            white-space: nowrap;
        }
        .header-links {
            text-align: center;
        }
        .header-links a {
            display: inline-block;
            text-align: center;
            margin-right: 10px;  /* 调整间距 */
        }
    """) as demo:
        gr.HTML(
            "Table Recognition Demo"
        )
        with gr.Row():  # 两列布局
            with gr.Tab("Options"):
                with gr.Column(variant="panel", scale=1):  # 侧边栏，宽度比例为1
                    img_input = gr.Image(label="Upload or Select Image", sources="upload", value="images/lineless3.jpg")

                    # 示例图片选择器
                    examples = gr.Examples(
                        examples=example_images,
                        examples_per_page=len(example_images),
                        inputs=img_input,
                        fn=lambda x: x,  # 简单返回图片路径
                        outputs=img_input,
                        cache_examples=False
                    )

                    table_engine_type = gr.Dropdown(table_engine_list, label="Select Recognition Table Engine",
                                                    value=table_engine_list[0])
                    small_box_cut_enhance = gr.Checkbox(
                        label="识别框切割增强(wiredV2,关闭避免多余切割，开启减少漏切割)",
                        value=True
                    )
                    remove_line = gr.Checkbox(
                        label="去除表格线(hzw)",
                        value=True
                    )
                    preproc = gr.Checkbox(
                        label="预处理(去噪、二值化等，效果不佳)(hzw)",
                        value=False
                    )
                    char_ocr = gr.Checkbox(
                        label="单字符OCR匹配",
                        value=True
                    )
                    rotate_adapt = gr.Checkbox(
                        label="表格旋转识别增强(wiredV2)",
                        value=False
                    )
                    col_threshold = gr.Slider(
                        label="同列x坐标距离阈值(wiredV2)",
                        minimum=5,
                        maximum=100,
                        value=5,
                        step=5
                    )
                    row_threshold = gr.Slider(
                        label="同行y坐标距离阈值(wiredV2)",
                        minimum=5,
                        maximum=100,
                        value=5,
                        step=5
                    )

                    # det_model = gr.Dropdown(det_models_labels, label="Select OCR Detection Model",
                    #                         value=det_models_labels[0])
                    # rec_model = gr.Dropdown(rec_models_labels, label="Select OCR Recognition Model",
                    #                         value=rec_models_labels[0])

                    run_button = gr.Button("Run")
                    gr.Markdown("# Elapsed Time")
                    elapse_text = gr.Text(label="")  # 使用 `gr.Text` 组件展示字符串
            with gr.Column(scale=2):  # 右边列
                # 使用 Markdown 标题分隔各个组件
                gr.Markdown("# Html Render")
                html_output = gr.HTML(label="", elem_classes="scrollable-container")
                gr.Markdown("# Table Boxes")
                table_boxes_output = gr.Image(label="")
                gr.Markdown("# OCR Boxes")
                ocr_boxes_output = gr.Image(label="")

        run_button.click(
            fn=process_image,
            inputs=[img_input, small_box_cut_enhance, table_engine_type, char_ocr, rotate_adapt, col_threshold, row_threshold, remove_line, preproc],
            outputs=[html_output, table_boxes_output, ocr_boxes_output, None, elapse_text]
        )

    demo.launch()


if __name__ == '__main__':
    main()
