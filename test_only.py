from ultralytics import YOLO
import cv2
import os
import wandb

wandb.init(project="yolov8")

model = YOLO("yolov8m.yaml") 

model = YOLO("runs/detect/train4/weights/last.pt")

wandb.watch(model)

results = model("datasets/pe_module_24_3_25/images/test/")

for idx, result in enumerate(results):
    wandb.log({
    # 'images': wandb.Image(results[0]),
    'prediction result': wandb.Image(result)
})
#   img_path = os.path.join("./results/test/yolov8m_default_lastpt_newdata/", f"img_{idx}.png")
#   cv2.imwrite(img_path, result.plot(font_size=0.1))
# # cv2.imshow("plot", plots)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# from ultralytics import YOLO
# import cv2
# model = YOLO("yolov8s.pt")

# results = model("./test.png")
# plots = results[0].plot()

# boxes = results[0].boxes

# for box in boxes :
#     print(box.xyxy.cpu().detach().numpy().tolist())
#     print(box.conf.cpu().detach().numpy().tolist())
#     print(box.cls.cpu().detach().numpy().tolist())