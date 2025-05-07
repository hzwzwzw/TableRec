import myapp
import os
import cv2
# myapp.main()
dataset_dir = "dataset"
result_dir = "result"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

timetot = 0
filetot = 0

files = os.listdir(dataset_dir)
files.sort()

for filename in files:
    file_path = os.path.join(dataset_dir, filename)
    if os.path.isfile(file_path):
        # print(filename)
        if "ss" in filename or True:
            complete_html, table_boxes_img, ocr_boxes_img, elapse, _ = myapp.process_image(file_path, True, None, None, True, 5, 5, None, False)
        else:
            complete_html, table_boxes_img, ocr_boxes_img, elapse, _ = myapp.process_image(file_path, True, None, None, True, 5, 5, None, True)
        # complete_html是一个html文本， 另外二者是图片
        # 将三个文件都保存到result_dir目录下
        with open(os.path.join(result_dir, filename + ".html"), "w") as f:
            f.write(complete_html)
        cv2.imwrite(os.path.join(result_dir, filename + "_table_boxes.jpg"), table_boxes_img)
        cv2.imwrite(os.path.join(result_dir, filename + "_ocr_boxes.jpg"), ocr_boxes_img)
        timetot += elapse[0]
        filetot += 1
        print(f"Processed {filename} in {elapse[0]:.2f} seconds, table det: {elapse[1]:.2f} seconds, ocr det: {elapse[2]:.2f} seconds, ocr preproc: {(elapse[3] - elapse[5]):.2f} seconds, ocr rec: {elapse[5]:.2f} seconds.")
print(f"Processed {filetot} files in {timetot:.2f} seconds, average {timetot/filetot:.2f} seconds per file.")