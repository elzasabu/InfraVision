# Training InfraVision on POP

Put the POP dataset at the repo root as `POP/` (`POP/{train,val,test}/JPEGImages` and
`POP/{train,val,test}/Annotations/{split}.json`). Then, from the repo root:

## 1. Super-resolution (SwinIR), on the laptop or any GPU

```bash
cd SR
python prepare_pop_dataset.py   # POP crops -> SR/dataset_pop/
python train_swinir.py          # -> SR/checkpoints_pop/swinir_direct/
cd ..
```

## 2. Person detection (YOLO11s)

```bash
python human_detection/datasets/convert_pop.py    # POP COCO json -> human_detection/datasets/pop/
python human_detection/datasets/make_pop_zip.py   # -> human_detection/pop_yolo_colab.zip
```

`make_pop_zip.py` adds the FLIR thermal model (`runs/thermal/colab_resumed/weights/best.pt`)
as the starting point when it exists. Pass another `.pt` path to use that instead, or `none`
to start from the standard `yolo11s.pt`.

Then train on a free GPU with one of the notebooks:

- **Colab:** upload the zip to `My Drive/Infravision/`, open `train_pop_yolo_colab.ipynb`,
  pick a T4 GPU and **Run all**.
- **Kaggle:** upload the zip as a dataset and run `train_pop_yolo_kaggle.ipynb`
  (steps are at the top of the notebook).

Download `pop_1024.zip` at the end and unzip it into `human_detection/runs/pop/pop_1024/`.

No GPU at all? `python human_detection/train.py` trains at 640 px into `runs/pop/baseline/`,
but it is slow on a CPU.

## 3. Run the demo

```bash
python integrate_infravision.py
```

It loads `human_detection/runs/pop/pop_1024/weights/best.pt`, a POP test image from
`human_detection/datasets/pop/images/test/` and the SwinIR checkpoint, and writes the results to
`integrated_outputs/`.
