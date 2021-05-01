# Table_Ocr_With_Paddle

This is the Table extraction model in which you just turn an image of a table and turned it into the csv. 

**This is not use to detect the table in the paper.**

![Example Image](example.png)


|Baseline        |BLEU-A  |BLEU-1|BLEU-2|BLEU-3|BLEU-4|ROUGE-LCSF1|FIELD8     |
|----------------|--------|------|------|------|------|-----------|-----------|
|ast-attendgru   |18.69   |37.13 |21.11 |14.27 |10.90 |49.75      |           |
|graph2seq       |18.61   |37.56 |21.27 |14.13 |10.63 |49.69      |           |
|code2seq        |18.84   |37.49 |21.36 |14.37 |10.95 |49.69      |           |
|BiLSTM+GNN-iLSTM|19.05   |37.70 |21.53 |14.59 |11.11 |55.74      |           |
|ConvGNNModels   |#of hops|BLEU-A|BLEU-1|BLEU-2|BLEU-3|BLEU-4     |ROUGE-LCSF1|
|code+gnn+dense  |2       |19.46 |38.71 |22.04 |14.86 |11.31      |56.07      |
|code+gnn+BiLSTM |2       |19.93 |39.14 |22.49 |15.31 |11.70      |56.08      |
|code+gnn+GRU    |1       |19.70 |38.15 |22.12 |15.22 |11.73      |57.15      |
|code+gnn+GRU    |2       |19.89 |39.01 |22.42 |15.28 |11.70      |55.78      |
|code+gnn+GRU    |3       |19.58 |38.48 |22.09 |15.01 |11.52      |56.14      |
|code+gnn+GRU    |5       |19.68 |38.89 |22.30 |15.09 |11.46      |55.81      |
|code+gnn+GRU    |10      |19.34 |38.68 |21.94 |14.73 |11.20      |55.10      |


# Dependent 
All codes modified from the PaddlePaddle OCR, need to first setup [paddlepaddle](https://github.com/PaddlePaddle/PaddleOCR)

[QUICK INSTALL](QuickStart.md)


# Model
Download Model from PP-OCR and put it into the inference file
- [Recognition model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar)
- [Detection model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar)
- [Direction classifier](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar)



```
./inference/
    ch_ppocr_mobile_v2.0_cls_infer/
    ch_ppocr_server_v2.0_det_infer/
    ch_ppocr_server_v2.0_rec_infer/
```


# Use

```
sh infer_table.sh
```

After pops out the waiting line `Extract Table From Image ("?"/"h" for help,"x" for exit)`

Just use your Screenshots tools to cut an image in the clipboard
and input enter. You will see the final result in the `./example.csv` and the screenshot as `pic.png`


OR use it with local image `--image_dir=''`

```
python3  tools/infer/predict_table.py --clipboard False --image_dir='pic.png' --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/"  --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/"  --use_angle_cls=True --use_space_char=True --use_gpu=False 



```
