#!/usr/bin/env bash
# should be executed in the projects root dir!
wavenet_model_url="https://drive.google.com/uc?export=download&id=0B3ILZKxzcrUyVWwtT25FemZEZ1k"
adapt_timit_url="https://drive.google.com/uc?export=download&id=1Df_wwFBfjM4gAmQO-Iv_himz7JoV3OQy"
mkdir -p ./train
if [ 3 -eq $(ls -al ./train/|grep model|wc -l) ] ;then 
    echo "Already downloaded: ${wavenet_model_url}";
else
    wget -c "${wavenet_model_url}" -O ./train.zip
    unzip ./train.zip -d ./train/
fi

mkdir -p ./adapt_timit
if [ 3 -eq $(ls -al ./adapt_timit/|grep model|wc -l) ] ;then
	echo "Already downloaded: ${adapt_timit_url}";
else
	wget -c "${adapt_timit_url}" -O ./adapt_timit.tar.xz
	tar xJf ./adapt_timit.tar.xz -C ./adapt_timit
	mv ./adapt_timit/*/* ./adapt_timit/
	rm -rf ./adapt_timit/train_fuck_timit
fi

