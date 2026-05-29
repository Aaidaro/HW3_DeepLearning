# Football Semantic Segmentation Homework

This project implements the requested U-Net segmentation pipeline and the YOLOv8n coco8 transfer-learning task using only paths relative to the project root.

## 1. Dataset placement

Place your downloaded datasets like this:

```text
homework_code/
└── data/
    └── dataset/
        ├── dataset1/
        │   ├── images/
        │   └── masks/
        └── dataset2/
            └── images/
```

Dataset 2 masks are discovered using filenames containing `___fuse`, for example:

```text
Frame 1 (1).JPG
Frame 1 (1).jpg___fuse.PNG
```

Dataset 2 fuse masks may have 4 channels. The fourth channel is the alpha/transparency channel, so the code drops it and uses only RGB for class remapping.

## 2. Unified classes

The model predicts 9 classes:

```text
Background, Player, Goalkeeper, Referee, Ball, Goal Bar, Advertisement, Audience, Staff
```

Field/Ground are mapped to Background. Team 1/Team 2 and Team A/Team B are merged into Player. Goalkeeper 1/2 and Goalkeeper A/B are merged into Goalkeeper.

## 3. Run data analysis

```bash
python -m scripts.main --mode analysis
```

Outputs:

```text
results/analysis/class_distribution_train_val.png
results/analysis/class_distribution_train_val.csv
```

The plot shows pixel percentages for each class after the 90/10 train/validation split. The split is done separately per dataset so both datasets appear in validation.

## 4. Train four U-Net cases

```bash
python -m scripts.main --mode baseline4
```

The four cases are:

1. bilinear upsample + skip connections
2. bilinear upsample + no skip connections
3. transposed convolution + skip connections
4. transposed convolution + no skip connections

Outputs include checkpoints, loss curves, mIoU curves, and a summary CSV:

```text
models/saved_models/*_best.pt
results/segmentation/<case_name>/loss_curves.png
results/segmentation/<case_name>/miou_curve.png
results/segmentation_summary.csv
```

## 5. BatchNorm and Focal Loss stages

After the baseline four cases finish:

```bash
python -m scripts.main --mode bn
python -m scripts.main --mode focal
```

`bn` selects the best previous architecture and retrains it with Conv+BN+ReLU. `focal` selects the best available result in the summary and retrains the same architecture using manually implemented focal loss.

You can run everything in sequence with:

```bash
python -m scripts.main --mode all
```

## 6. Qualitative comparison of several checkpoints

Example:

```bash
python -m scripts.evaluate \
  --checkpoints \
  models/saved_models/bilinear_skip_nobn_cross_entropy_best.pt \
  models/saved_models/bilinear_noskip_nobn_cross_entropy_best.pt \
  models/saved_models/transpose_skip_nobn_cross_entropy_best.pt \
  models/saved_models/transpose_noskip_nobn_cross_entropy_best.pt
```

Outputs:

```text
results/evaluation/checkpoint_metrics.csv
results/evaluation/qualitative_comparison.png
```

## 7. YOLOv8n coco8 task

Install dependencies, then run:

```bash
python -m scripts.main --mode yolo
```

This loads `yolov8n.pt`, trains on `coco8.yaml` for 50 epochs, freezes the first 10 layers, saves Ultralytics plots, and predicts on one validation image.

Outputs:

```text
results/yolo/coco8_yolov8n_freeze10/results.png
results/yolo/validation_prediction/
```

## 8. Notes for the report

Cropping is not necessary in this implementation because each 3x3 convolution uses padding=1, so the convolutional layers preserve H and W. For 256x256 input, max-pooling and upsampling restore matching spatial sizes. A size-matching fallback is included only for robustness to odd image sizes.

Class imbalance is expected in soccer segmentation: Background/Ground/Field usually dominates the pixels, while Ball, Goal Bar, Referee, and Staff may be very small. This can cause the model to over-predict frequent classes and ignore small classes, producing high pixel accuracy but low mIoU for rare classes. Two common solutions are class-aware losses such as weighted cross entropy or focal loss, and data-level strategies such as targeted oversampling/cropping around rare classes or stronger augmentation of images containing small objects.
