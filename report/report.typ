// Formal project report (max. 5 pages). Build: python scripts/build_report_pdf.py

#let title = "Underwater Fish and Marine Animal Detection with YOLOv8"
#let author = "Emre Yoleri"
#let date = "October 2026"

#set document(title: title, author: author)
#set page(paper: "a4", margin: (x: 2.5cm, top: 2.5cm, bottom: 2.3cm),
  footer: context align(center, text(size: 9pt, counter(page).display("1"))))
#set text(font: "Times New Roman", size: 11pt, lang: "en", region: "us")
#set par(justify: true, leading: 0.62em, spacing: 0.95em, first-line-indent: 0pt)
#set heading(numbering: "1.1.")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)
#show heading: set block(above: 1.2em, below: 0.7em)
#set math.equation(numbering: "(1)")
#set figure(gap: 0.6em)
#show figure: set block(above: 1em, below: 1em)
#show figure.caption: set text(size: 9.5pt)
#show figure.caption: it => [*#it.supplement #context it.counter.display(it.numbering).* #it.body]
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: (x, y) => (top: if y <= 1 { 0.6pt } else { 0pt }, bottom: 0.6pt), inset: (x: 5pt, y: 3.2pt))
#show table: set text(size: 9.5pt)
#show table.cell.where(y: 0): strong

// ---------------------------------------------------------------- title block
#align(center)[
  #text(size: 15pt, weight: "bold")[#title]
  #v(0.5em)
  #text(size: 11pt)[#author]
  #v(0.1em)
  #text(size: 10pt, style: "italic")[Technical Report · #date]
]
#v(0.6em)
#line(length: 100%, stroke: 0.5pt)
#block(inset: (x: 0.6cm))[
  #set text(size: 10pt)
  *Abstract.* Automatic detection of fish and other marine animals in underwater footage is needed for
  biodiversity monitoring and aquarium and fisheries analytics, but underwater images are degraded by a
  blue–green colour cast, haze, blur and low light. This report presents a detection system based on the
  YOLOv8s one-stage detector, fine-tuned from COCO weights on the public Roboflow Aquarium dataset with
  seven classes (fish, jellyfish, penguin, puffin, shark, starfish and stingray). The training set is
  doubled with offline, box-preserving augmentations that simulate underwater degradation (colour cast,
  turbidity, blur and low light), and an OpenCV preprocessing module (red-channel compensation, white
  balance and CLAHE) is provided for inference. After a short training schedule of 10 epochs on a laptop
  GPU, the model reached a precision of 0.752, a recall of 0.654, an mAP50 of 0.727 and an mAP50-95 of
  0.414 on the held-out test set (63 images, 582 objects), and processed a 1280×720 aquarium video at
  about 15 frames per second.

  *Keywords:* object detection, YOLOv8, underwater imaging, fish detection, data augmentation, mAP.
]
#line(length: 100%, stroke: 0.5pt)

// ---------------------------------------------------------------- body
= Introduction

Cameras are increasingly used to monitor marine life in reefs, fish farms and aquariums, and the amount
of video they produce is too large for manual annotation @salman2020fish @villon2018coral. Underwater images are, however, difficult for detectors: water
absorbs red light and gives the scene a blue–green cast, suspended particles scatter light and reduce
contrast @akkaynak2018, and moving animals and cameras cause motion blur. In addition, animals are often
small, partly occluded and densely grouped.

One-stage detectors of the YOLO family @redmon2016yolo predict all boxes in a single network pass and
are fast enough for real-time video. The objective of this work is to (i) fine-tune a YOLOv8 detector
@jocher2023yolov8 for seven classes of marine animals, (ii) add underwater-specific data augmentation
and preprocessing built with OpenCV and NumPy, (iii) evaluate the detector with standard metrics —
precision, recall and mean average precision (mAP) — overall and per class, and (iv) measure its speed
on real video.

= Methodology

== Dataset

The Roboflow Aquarium dataset, obtained from the public Roboflow 100 benchmark @ciaglia2022rf100,
contains photographs taken in two aquariums in the United States, annotated with bounding boxes for
seven classes. The COCO-format export was converted to YOLO format (normalised centre, width and
height), and Roboflow's empty super-category was removed. The official split was kept: 448 training,
127 validation and 63 test images. @tab-data gives the number of annotated objects per class. The data
is strongly imbalanced: fish account for about half of all boxes, while starfish and stingray appear
only 11 and 15 times in the test set.

#figure(
  table(
    columns: (1.3fr, 0.8fr, 0.9fr, 0.9fr, 0.8fr, 0.7fr, 0.8fr, 0.8fr, 0.7fr),
    align: (left, center, center, center, center, center, center, center, center),
    table.header([Split], [Images], [Fish], [Jellyfish], [Penguin], [Puffin], [Shark], [Starfish], [Stingray]),
    [Train (+aug.)], [896], [1964], [385], [330], [175], [259], [78], [136],
    [Validation], [127], [459], [155], [104], [74], [57], [27], [33],
    [Test], [63], [249], [154], [82], [35], [36], [11], [15],
  ),
  caption: [Number of images and annotated objects per split. The training split contains each of the
  448 original images and one augmented copy.],
) <tab-data>

== Underwater Augmentation and Preprocessing

To make the detector robust to the variation found under water, one augmented copy of every training
image was generated offline (fixed random seed). Each copy is horizontally flipped with probability 0.5
(boxes are mirrored accordingly) and then receives one or two randomly chosen photometric transforms:
a *colour cast* that scales the B, G and R channels by random gains in [1.0, 1.25], [0.95, 1.2] and
[0.6, 0.95]; *turbidity*, which blends the image with a random haze colour $A$ as
$I' = t I + (1 - t) A$ with $t in [0.6, 0.9]$, a simplified form of the scattering model
@akkaynak2018; Gaussian or horizontal motion *blur* with a 3–7 pixel kernel; and *low light*, a gamma
curve with $gamma in [1.4, 2.0]$ followed by Gaussian noise. Photometric transforms do not move objects,
so the labels stay valid. During training, Ultralytics applies its own online augmentation on top
(mosaic @bochkovskiy2020yolov4, HSV jitter, scaling and translation).

For inference, an optional OpenCV preprocessing step is available: red-channel compensation from the
green channel @ancuti2018, Gray-World white balance @buchsbaum1980 and CLAHE @zuiderveld1994 on the
$L^*$ channel of CIELAB (clip limit 2.0, 8×8 tiles). @fig-aug shows an example of each transform.

#figure(
  image("/report/figures/augmentation.jpg", width: 64%),
  caption: [Underwater augmentations used for training (first five panels) and the optional colour
  and contrast correction used before inference (last panel).],
) <fig-aug>

== Model and Training

YOLOv8s (11.1 M parameters, 28.4 GFLOPs) was chosen as a compromise between accuracy and speed. It uses
a CSP-based backbone, a feature-pyramid neck and an anchor-free decoupled head; boxes are trained with
CIoU and distribution focal loss @li2020gfl and classes with binary cross-entropy. The network was
initialised with weights pre-trained on COCO @lin2014coco and fine-tuned at an input size of 640×640
with a batch size of 16 on an Apple M3 Pro GPU (MPS backend). The optimiser was AdamW
@loshchilov2019adamw with a learning rate of $9.1 times 10^(-4)$ (selected automatically by
Ultralytics), weight decay $5 times 10^(-4)$ and a cosine learning-rate schedule.

Because of limited compute time, training was done in two short stages. Stage 1 ran 4 epochs of a
planned 50-epoch schedule, including the 3-epoch learning-rate warm-up. Stage 2 continued from the
last stage-1 weights for 6 epochs without warm-up, with the cosine schedule decaying the learning rate
to near zero and mosaic augmentation switched off for the last 3 epochs. The weights with the best
validation fitness were kept.

== Video Inference and Evaluation

For video, each frame is optionally preprocessed, passed to the detector (confidence 0.25, NMS IoU 0.5),
annotated and written as H.264 MP4; the same detector powers a Streamlit application for images, videos
and webcam input. Latency was measured from preprocessing to the end of non-maximum suppression; the
end-to-end frame rate also includes decoding, drawing and encoding.

Detection quality was measured with the standard metrics @everingham2010voc @lin2014coco. A prediction
is a true positive if its IoU with an unmatched ground-truth box of the same class exceeds a threshold.
Precision $P = "TP" \/ ("TP" + "FP")$ and recall $R = "TP" \/ ("TP" +
"FN")$ are reported at the confidence that maximises $F_1 = 2 P R \/ (P + R)$. The average precision
(AP) is the area under the precision–recall curve; mAP50 is the mean AP over classes at IoU 0.5, and
mAP50-95 averages it over IoU thresholds 0.5:0.05:0.95, which also rewards precise localisation
(evaluation used confidence 0.001 and NMS IoU 0.6).

= Results

== Training

@fig-train shows the losses and validation mAP per epoch. In stage 1 the validation mAP50 stayed at
0.33–0.42 during the learning-rate warm-up. At the start of stage 2 it dropped to 0.22,
because the resumed run started directly at the full learning rate without warm-up, but it then rose in
every epoch to 0.693 at epoch 10 while all losses decreased. The curve was still rising at the end,
which shows that the model had not converged.

#figure(
  image("/report/figures/training_curves.png", width: 100%),
  caption: [Training and validation losses (left) and validation mAP (right) over the two training
  stages. The dotted line marks the start of stage 2.],
) <fig-train>

== Detection Accuracy

@tab-results summarises the results. On the test set the detector reached $P = 0.752$, $R = 0.654$
($F_1 = 0.699$), mAP50 = 0.727 and mAP50-95 = 0.414. The test scores are slightly above the validation
scores, so there is no sign of overfitting to the split used for model selection.

#figure(
  table(
    columns: (2.3fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center),
    table.header([Split], [Precision], [Recall], [F1], [mAP50], [mAP50-95]),
    [Validation (127 img., 909 obj.)], [0.747], [0.646], [0.693], [0.702], [0.403],
    [Test (63 img., 582 obj.)], [*0.752*], [*0.654*], [*0.699*], [*0.727*], [*0.414*],
  ),
  caption: [Overall detection results of YOLOv8s.],
) <tab-results>

#grid(
  columns: (1.08fr, 1fr),
  gutter: 0.5cm,
  align: horizon,
  [#figure(
    table(
      columns: (1.25fr, 0.8fr, 0.8fr, 0.9fr, 1fr),
      align: (left, center, center, center, center),
      table.header([Class], [P], [R], [mAP50], [mAP50-95]),
      [Fish], [0.790], [0.614], [0.744], [0.432],
      [Jellyfish], [0.783], [0.799], [0.847], [0.519],
      [Penguin], [0.759], [0.622], [0.718], [0.313],
      [Puffin], [0.812], [0.371], [0.524], [0.197],
      [Shark], [0.794], [0.750], [0.815], [0.491],
      [Starfish], [*0.907*], [*0.886*], [*0.922*], [*0.576*],
      [Stingray], [0.416], [0.533], [0.520], [0.367],
      table.hline(stroke: 0.4pt),
      [All], [0.752], [0.654], [0.727], [0.414],
    ),
    caption: [Per-class results on the test set.],
  ) <tab-class>],
  [#figure(
    image("/runs/eval/y8s_raw_test/BoxPR_curve.png", width: 100%),
    caption: [Precision–recall curves on the test set.],
  ) <fig-pr>],
)

Per-class results (@tab-class, @fig-pr) differ strongly. Starfish (mAP50 = 0.922), jellyfish (0.847)
and sharks (0.815) are detected best: they have distinctive shapes and, in the case of jellyfish, high
contrast against the blue background. The weakest classes are puffin (0.524) and stingray (0.520).
Puffins are small and often stand in dense groups on rocks above the water, which gives a high
precision (0.812) but the lowest recall (0.371). Stingrays are flat and seen from many angles, often
lying on the sand with little contrast; with only 136 training boxes (including augmented copies) and 15
test boxes, their scores are also the least reliable. Fish, the largest class, reach a high precision
(0.790) but a moderate recall (0.614), mainly because many small and overlapping fish in schools are
missed.

== Qualitative Results and Video Speed

@fig-qual compares predictions with the ground truth on four test images and shows two frames of the
processed video. Sharks, penguins and jellyfish are localised accurately; the misses are mostly small
or partly hidden animals, and one small false jellyfish box appears in the third image. In the puffin
image, the swimming bird receives two overlapping boxes, one of them much too large. In the aquarium video, which comes
from a different aquarium (Aquarium of Genoa, Wikimedia Commons, CC BY-SA 4.0), clownfish in an anemone
are found with moderate confidence, showing that the model generalises to new scenes, although some
fish are missed and the confidence is often below 0.5.

#figure(
  image("/report/figures/qualitative.jpg", width: 94%),
  caption: [Ground truth (top) and YOLOv8s predictions (middle, confidence ≥ 0.25) on four test images,
  and two frames of the annotated aquarium video (bottom).],
) <fig-qual>

The video has 1226 frames at 1280×720 pixels. The mean latency per frame was 65.9 ms (95th percentile
77.8 ms), i.e. 15.2 frames per second for the model and 14.2 frames per second end to end, on the
Apple M3 Pro. On average 3.8 objects were detected per frame; 4577 of the 4617 detections were labelled
as fish, and the remaining 40 were starfish, jellyfish, puffin and shark labels; since the clip shows
mainly reef fish, corals and anemones, most of these are probably misclassifications. On the 640×640 test images,
the network itself needed only 6.9 ms per image (batch 16), so a large part of the video latency is
probably overhead from per-frame resizing, a batch size of one and host–GPU transfer.

= Discussion

The results show that a small, COCO-pretrained one-stage detector can be adapted to underwater scenes
with very little training: after only 10 epochs, about 17 minutes on a laptop, it reached an mAP50 of
0.727 on unseen test images and ran at about 15 frames per second on high-definition video. The
per-class analysis shows that accuracy depends more on object size, crowding and the number of
training examples than on the underwater colour cast itself: large or high-contrast animals (sharks,
jellyfish, starfish) are detected well, while small, crowded objects (puffins, schooling fish) mainly
suffer from low recall. The gap between mAP50 and mAP50-95 (0.727 vs. 0.414) also shows that boxes are
usually found but are not always tightly localised, especially for flat stingrays and small puffins.

The work has several limitations. First, the model was not trained to convergence: the validation mAP
was still rising by about 0.04 per epoch at the end, and the training was split into two stages with a learning-rate
jump at the restart. Second, the dataset is small and imbalanced, and the per-class test scores for
starfish and stingray rest on only 11 and 15 objects. Third, the effect of the underwater augmentation
and of the CLAHE preprocessing was not measured separately, because only one model (trained on raw
images with augmentation) was evaluated; the preprocessing is implemented but its benefit is not
proven. Finally, all images come from aquariums, whose water is clearer than the open sea.

= Conclusion

A YOLOv8s-based system for detecting seven classes of marine animals was developed with Python, OpenCV
and Ultralytics, including underwater-specific augmentation, optional colour and contrast correction,
video inference and a Streamlit application. On the held-out test set of the Roboflow Aquarium dataset
it achieved a precision of 0.752, a recall of 0.654, an mAP50 of 0.727 and an mAP50-95 of 0.414, and it
processed 1280×720 video at about 15 frames per second on a laptop GPU. Future work includes a full
training schedule of 50 or more epochs, an ablation that compares training with and without the
underwater augmentation and CLAHE preprocessing, a larger input size or tiled inference for small
objects, and the addition of object tracking to count individual animals across video frames.

#v(0.3em)
#{
  set text(size: 9.5pt)
  set par(leading: 0.5em, spacing: 0.55em)
  show heading: set block(above: 1.2em, below: 0.7em)
  bibliography("references.yml", title: [References], style: "ieee")
}
