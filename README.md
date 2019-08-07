# WaveNet plus Source Speration



## WaveNet Speech Recognition

To arrpa phonemes

### Usage

1. Install required python libraries and corpus datasets through `make install`
2. Download pretrained models through  ``make data``);
3. Use ``make test`` to test sample audio.


Adapted model on TIMIT: [pretrained model download](https://drive.google.com/uc?export=download&id=1Df_wwFBfjM4gAmQO-Iv_himz7JoV3OQy)

### References

[CMU-Sphinx](https://github.com/cmusphinx)

[pocketsphinx for other languages](http://depado.markdownblog.com/2015-05-13-tutorial-on-pocketsphinx-with-python-3-4)

[wavnet-speech-recognition](https://github.com/buriburisuri/speech-to-text-wavenet): pre-trained model [here](https://drive.google.com/uc?export=download&id=0B3ILZKxzcrUyVWwtT25FemZEZ1k), need to modify sugartensor `__init__.py` in order to be compatible with `tensorflow >=1.11`



## Source sepration

### Dataset

[MIR-1K](https://sites.google.com/site/unvoicedsoundseparation/mir-1k) dataset

### Training

Set the dataset and checkpoint paths at config.py and run
```
python train_mir_1k.py
```
for MIR-1K dataset

### Evaluation

Run

```
python eval_mir_1k.py
```
for MIR-1K dataset

### Trained models

These are the checkpoint files for each dataset to reproduce the results on the paper.

[MIR-1K](https://www.dropbox.com/s/6759yx0zqer316f/mir_1k_checkpoints.zip?dl=0) (15000 training steps)

### References

[Music Source Separation Using Stacked Sourglass Networks](https://www.dropbox.com/s/w17nb9oqe7q5b8p/ISMIR18-sourceSep.pdf?dl=0)


