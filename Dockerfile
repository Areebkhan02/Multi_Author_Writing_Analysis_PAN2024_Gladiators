#docker build -t pan24-gladiators .
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip3 install pandas jupyterlab docker datasets transformers

RUN huggingface-cli download areebkhan02/Multi-author-writing-style-analysis-PAN2024

RUN ln -s /root/.cache/huggingface/hub/models--areebkhan02--Multi-author-writing-style-analysis-PAN2024/snapshots/72f7c97300816aebaf14ca18e95ad12bc677700e/ /models

RUN python3 -c 'import transformers; transformers.RobertaModel.from_pretrained("roberta-base"); transformers.RobertaTokenizer.from_pretrained("roberta-base");'

COPY mySolution.py /

ENTRYPOINT [ "/mySolution.py", "--input", "$inputDataset", "--output", "$outputDir" ]
