# Traffic Sign Detection Pipeline

This project detects and identifies traffic signs in real road scenes. It works in two stages: a YOLO26n detector finds where the signs are, then the CNN classifier I built in a previous project tells which sign each one is, out of 43 classes. When the classifier is not confident enough, the sign is reported as "unknown" instead of guessed.

On the test set, the full pipeline correctly names 91% of the signs and shows a wrong sign for none of them.

## How the pipeline works

```
Road scene (1360x800)
  -> YOLO26n detector (one class: "sign")
  -> crop each detected box and resize it to 32x32
  -> CNN classifier (43 GTSRB classes)
  -> confidence below 0.8? report "unknown sign"
  -> labeled boxes
```

I chose to train the detector on a single class instead of all 43. Some sign types appear only a handful of times in the detection dataset, which is far too few to learn from. A single "sign" class gives the detector every example at once, and the hard part, telling 43 sign types apart, goes to my classifier, which was trained on about 39,000 images.

## Dataset

I used the German Traffic Sign Detection Benchmark (GTSDB): 900 road images containing 1,213 signs.

Before training, I inspected it and found two things that shaped the rest of the project:

- About 18% of the images contain no sign at all. I kept them as background examples, so the model learns what "nothing here" looks like.
- Signs are small. They range from 16 to 127 pixels wide, and a third of them are under 32 pixels.

I converted the labels to YOLO format myself and split the images 70/15/15 into train, validation and test sets (629 / 134 / 137). The split is done by image, so signs from the same scene never end up in two sets, and it is stratified so each set keeps the same share of background images. I checked the result with sanity checks on the files and by drawing the converted boxes back onto the images.

Every decision in this project (resolution, thresholds) was made on the validation set. The test set was used only for the final evaluations.

## Part 1: The detector

### Choosing the input resolution

At YOLO's default input size of 640, every image is shrunk to less than half its width, and the smallest signs end up around 7 pixels wide. I trained two models that differ only in input size, to see how much this mattered.

Results on the validation set:

| Metric    | 640   | 1280  |
|-----------|-------|-------|
| Precision | 0.965 | 0.994 |
| Recall    | 0.902 | 0.977 |
| mAP50     | 0.962 | 0.993 |
| mAP50-95  | 0.808 | 0.874 |

To check where the improvement came from, I split the validation signs by size and compared both models (confidence threshold 0.25):

| Signs        | 640             | 1280            |
|--------------|-----------------|-----------------|
| Under 32 px  | 53/69 (0.768)   | 68/69 (0.986)   |
| 32 px and up | 112/115 (0.974) | 114/115 (0.991) |

At 640, the model missed almost one small sign in four, while finding nearly all the large ones. Of the 17 extra signs found at 1280, 15 are small. Higher resolution did not make the model better in general; it fixed the small-sign problem specifically. I kept the 1280 model.

### Looking at the errors

The metrics say how often the model is wrong, but not why, so I matched every prediction to the true boxes myself and inspected each error. On the validation set, the 1280 model found 182 signs, missed 2, and produced 7 false positives.

- Both missed signs were overexposed. Strong light had washed out their red borders, which seem to be the model's main visual cue.
- Five of the seven false positives turned out to be real signs that simply have no label in the dataset, either because they are very small or because they are not among the 43 classes (a "no stopping" sign, for example). Only two were genuine mistakes: round and triangular shapes on buildings.

### Detector test results

Evaluated once on the test set (137 images, 200 signs): precision 0.970, recall 0.972, mAP50 0.990, mAP50-95 0.857. These are slightly below the validation scores, which is expected since the best checkpoint was selected on validation. On the test set, the model finds 71 of 74 small signs and 125 of 126 larger ones.

## Part 2: Connecting the classifier

### The classifier alone, with perfect boxes

To separate the two stages, I first cropped the validation signs using the human-drawn boxes and gave them to the classifier, prepared exactly like its training test images (resized to 32x32, no other change). It named 178 of 184 correctly (96.7%). That is the best the pipeline could do with a perfect detector.

All six mistakes fell into the two error clusters I had already found on GTSRB: speed limit digits (120 read as 100, 80 as 60) and triangular warning signs that differ only by a small pictogram. Five of the six were large signs, so the cause is the classifier's 32x32 input, which shrinks away fine details, not the size of the sign.

### The full pipeline

With YOLO providing the boxes, 175 of 184 validation signs were found and correctly named (95.1%). The 9 errors break down as:

| Source of error                                   | Signs |
|---------------------------------------------------|-------|
| Classifier mistakes, even with perfect boxes      | 6     |
| Signs missed by the detector                      | 1     |
| Differences between YOLO's boxes and human boxes  | 2     |

So most errors come from the classifier, not the detector. Small differences in the crop also flipped some of the classifier's borderline answers in both directions: YOLO's boxes broke 3 signs and fixed 1.

### The "unknown sign" rule

A wrong sign shown to the driver is more dangerous than "unknown": the driver or another source, such as map data, can take over when the system admits it does not know. So I added a rule: if the classifier's confidence is below a threshold, the sign is reported as "unknown."

I chose the threshold on the validation set by counting, for each value, the dangerous outputs prevented (wrong signs and false detections given a name) against the correct answers lost:

| Threshold | Dangerous outputs | Correct answers lost |
|-----------|-------------------|----------------------|
| none      | 14                | 0                    |
| 0.6       | 10                | 0                    |
| 0.8       | 4                 | 3                    |
| 0.9       | 2                 | 11                   |

I chose 0.8: it removes most dangerous outputs for a small cost, and above it the cost rises quickly.

### Final pipeline results (test set)

Evaluated once on the test set, with every setting fixed in advance (200 signs):

| Outcome          | Signs | Share |
|------------------|-------|-------|
| Correctly named  | 182   | 91.0% |
| Wrong sign shown | 0     | 0.0%  |
| "Unknown"        | 14    | 7.0%  |
| Missed           | 4     | 2.0%  |

The correctly named share is lower than on validation (93.5%), as expected since the threshold was chosen there. No wrong sign was shown for any real sign: every time the classifier would have been wrong, it was unsure enough for the rule to catch it.

## Limitations

- False detections are the main remaining weakness. Of the 17 false detections on the test set, 8 were still given a sign name. Many of them are real signs outside the 43 classes, and the classifier is overconfident on things it has never seen, so a confidence threshold catches only some of them.
- Some classifier mistakes are made with high confidence (120 read as 100 at 0.94), so no threshold can catch them without losing many correct answers.
- Overexposed signs are hard for both stages, since the red border is the main visual cue.
- The classifier's 32x32 input limits how well it can read digits and small pictograms.
- Everything was trained and tested on German signs only, with small validation and test sets (184 and 200 signs), so the exact percentages carry some uncertainty.

## Project structure

```
notebook/
  01_inspect_gtsdb.ipynb          dataset statistics
  02_prepare_yolo_dataset.ipynb   conversion, split and sanity checks
  03_train_yolo_colab.ipynb       training at 640 and 1280 on Google Colab
  04_error_analysis.ipynb         detector errors, 640 vs 1280 by sign size, detector test results
  05_pipeline.ipynb               classifier integration, "unknown" rule, final pipeline evaluation
src/
  classifier_model.py             the classifier's architecture
  detection_utils.py              IoU and matching of predictions to true boxes
models/
  yolo26n_1280_best.pt            the detector used in the pipeline
  yolo26n_640_best.pt             the 640 baseline, for comparison
  gtsrb_cnn_v4_bn.pth             the classifier
```

## Next steps

- [ ] Give the classifier an "other sign" class, so it can recognize signs outside its 43 classes
- [ ] Try a higher input resolution for the classifier, to reduce the digit and pictogram confusions
- [ ] Test adding a margin around YOLO's boxes before cropping, to better match how the classifier's training images were framed
- [ ] Run the pipeline on video

## Tools

Python, PyTorch, torchvision, Ultralytics YOLO26, Pillow, Google Colab (T4 GPU)

The classifier comes from my earlier project: [traffic-sign-classifier](https://github.com/yessin-hamdi/traffic-sign-classifier)