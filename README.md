# WaveNet Speech Recognition

To arrpa phonemes

## Usage

1. Install required python libraries and corpus datasets through `make install`
2. Download pretrained models through  ``make data``);
3. Use ``make test`` to test sample audio.


Adapted model on TIMIT: [pretrained model download](https://drive.google.com/uc?export=download&id=1Df_wwFBfjM4gAmQO-Iv_himz7JoV3OQy)



## References

[CMU-Sphinx](https://github.com/cmusphinx)

[pocketsphinx for other languages](http://depado.markdownblog.com/2015-05-13-tutorial-on-pocketsphinx-with-python-3-4)

[wavnet-speech-recognition](https://github.com/buriburisuri/speech-to-text-wavenet): pre-trained model [here](https://drive.google.com/uc?export=download&id=0B3ILZKxzcrUyVWwtT25FemZEZ1k), need to modify sugartensor `__init__.py` in order to be compatible with `tensorflow >=1.11`


