#!/usr/bin/env bash
# should be executed in the projects root dir!
model_dirs=("train" "adapt_timit")
model_urls=("https://drive.google.com/uc?export=download&id=0B3ILZKxzcrUyVWwtT25FemZEZ1k"
	"https://drive.google.com/uc?export=download&id=1fYr_Z7lUB_j0GW8SgWWupd0A9bM-WaQs"
	)

#wavenet_model_url="https://drive.google.com/uc?export=download&id=0B3ILZKxzcrUyVWwtT25FemZEZ1k"
#adapt_timit_url="https://drive.google.com/uc?export=download&id=1fYr_Z7lUB_j0GW8SgWWupd0A9bM-WaQs"

for i in ${!model_dirs[@]}; do
	d=${model_dirs[$i]}
	mkdir -p ./${d}
	if [ 3 -eq $(ls -al ./${d}/|grep model|wc -l) ] ;then 
	    echo "Already downloaded: ${d}.zip";
	else
	    wget -c ${model_urls[$i]} -O ./${d}.zip
	    unzip ./${d}.zip -d ./${d}/
	fi
done
