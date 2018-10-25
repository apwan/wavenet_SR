
PY_EXE   =python3#
PIP_EXE  =pip3#


all: install data test
	@echo "Finished"

.PHONY: install data pipreqs

# installing requirements
install:
	-$(PIP_EXE) -q install -U -r ./requirements.txt
	@$(PY_EXE) -c 'from utils import data_install; data_install("cmudict"); data_install("timit");'

pipreqs:
	@which pipreqs || $(PIP_EXE) install pipreqs
	@pipreqs --print ./ | sort -o ./requirements.txt


data:
	-@./download_pretrained.sh



test:
	@./wavenet.py

