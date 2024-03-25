from ultralytics import YOLO
import wandb


def main():
    wandb.init(project="yolov8")

    model = YOLO('yolov8m.yaml')

    model = YOLO('/results/middle_default_pe325/yolov8m.pt')

    wandb.watch(model)

    results = model.train(data='pe_module.yaml', epochs=100)

    results = model.val()


if __name__ == '__main__':
    main()

