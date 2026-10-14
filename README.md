\# Traffic Sign Detection Pipeline



This project detects and identifies traffic signs in real road scenes. It works in two stages: a YOLO26n detector finds where the signs are, then the CNN classifier I built in a previous project tells which sign each one is, out of 43 classes.



This is a work in progress. The detector is trained and analyzed, and connecting the classifier is the next step.



\## How the pipeline works



```

Road scene (1360x800)

&#x20; -> YOLO26n detector (one class: "sign")

&#x20; -> crop each detected box and resize it to 32x32

&#x20; -> CNN classifier (43 GTSRB classes)

&#x20; -> labeled boxes

```



I chose to train the detector on a single class instead of all 43. Some sign types appear only a handful of times in the detection dataset, which is far too few to learn from. A single "sign" class gives the detector every example at once, and the hard part, telling 43 sign types apart, goes to my classifier, which was trained on about 39,000 images.



\## Dataset



I used the German Traffic Sign Detection Benchmark (GTSDB): 900 road images containing 1,213 signs.



Before training, I inspected it and found two things that shaped the rest of the project:



\- About 18% of the images contain no sign at all. I kept them as background examples, so the model learns what "nothing here" looks like.

\- Signs are small. They range from 16 to 127 pixels wide, and a third of them are under 32 pixels.



I converted the labels to YOLO format myself and split the images 70/15/15 into train, validation and test sets (629 / 134 / 137). The split is done by image, so signs from the same scene never end up in two sets, and it is stratified so each set keeps the same share of background images. I checked the result with sanity checks on the files and by drawing the converted boxes back onto the images.



\## Choosing the input resolution



At YOLO's default input size of 640, every image is shrunk to less than half its width, and the smallest signs end up around 7 pixels wide. I trained two models that differ only in input size, to see how much this mattered.



Results on the validation set:



| Metric    | 640   | 1280  |

|-----------|-------|-------|

| Precision | 0.965 | 0.994 |

| Recall    | 0.902 | 0.977 |

| mAP50     | 0.962 | 0.993 |

| mAP50-95  | 0.808 | 0.874 |



The 1280 model misses about 4 signs out of 184, compared with about 18 at 640, and its boxes fit more tightly. I kept it.
To check where the improvement came from, I split the validation signs by size and compared both models (confidence threshold 0.25):

| Signs        | 640           | 1280          |
|--------------|---------------|---------------|
| Under 32 px  | 53/69 (0.768) | 68/69 (0.986) |
| 32 px and up | 112/115 (0.974) | 114/115 (0.991) |

At 640, the model missed almost one small sign in four, while finding nearly all the large ones. Of the 17 extra signs found at 1280, 15 are small. Higher resolution did not make the model better in general; it fixed the small-sign problem specifically.




\## Looking at the errors



The metrics say how often the model is wrong, but not why, so I matched every prediction to the true boxes myself and inspected each error. With a confidence threshold of 0.25, the 1280 model found 182 signs, missed 2, and produced 7 false positives.



\- Both missed signs were overexposed. Strong light had washed out their red borders, which seem to be the model's main visual cue.

\- Five of the seven false positives turned out to be real signs that simply have no label in the dataset, either because they are very small or because they are not among the 43 classes (a "no stopping" sign, for example). Only two were genuine mistakes: round and triangular shapes on buildings.



This points to a real risk for the full pipeline. The detector finds signs the classifier has never seen, and the classifier will be forced to pick one of its 43 classes anyway. So the next step includes an "unknown sign" option, used when the classifier is not confident enough.



\## Project structure



```

notebook/

&#x20; 01\_inspect\_gtsdb.ipynb          dataset statistics

&#x20; 02\_prepare\_yolo\_dataset.ipynb   conversion, split and sanity checks

&#x20; 03\_train\_yolo\_colab.ipynb       training at 640 and 1280 on Google Colab

&#x20; 04\_error\_analysis.ipynb         matching predictions to true boxes, inspecting errors

models/

&#x20; yolo26n\_1280\_best.pt            the trained detector

```



\## Next steps



\- \[x] Compare recall on small signs between the 640 and 1280 models

\- \[ ] Run the final evaluation on the test set

\- \[ ] Connect the classifier, with an "unknown sign" option

\- \[ ] Run the pipeline on video



\## Tools



Python, PyTorch, Ultralytics YOLO26, Pillow, Google Colab (T4 GPU)



The classifier comes from my earlier project: \[traffic-sign-classifier](https://github.com/yessin-hamdi/traffic-sign-classifier)

