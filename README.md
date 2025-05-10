## Data prep and Qwen model finetuning
1. Refer to the notebooks/DataPrepAndDownload.ipynb for data downloading the images and notebooks/Qwen2.5.ipynb for unsloth based
LORA finetune of Qwen2-VL-2B-Instruct model.
2. Chosen Qwen2-VL-2B-Instruct model for its size (4.2G), performance and ability to detect Chinese characters along with English.
3. The LORA adapter weights are stored in the final_chkpt folder.
4. Given the model size and its performance on varoius benchmarks, its a perfect fit for the usecase.
5. Able to finetune the model in less than 24GB of VRAM at 16bit precision instead of 4 bit as I wanted to optimize for accuracy.
6. In the interest of time I have finetuned with 50_000 samples for 1 epoch. This is given time constraints of my busy schedule. 
7. Ideally I would have finetuned longer with more  examples escially for Chinese characters and descriptions.
8. Overall I hope the methodology and approach is clear.

## Other considerations
1. To make inference faster I always use VLLM. I have not added it to the project as it is out of scope and also needs to run on GPU.
2. As mentioned above, due to lack of time I did not cherry pick examples for image description accuracy imporvements and formating of english `wordsInMark`. This is left for future work.
3. I have also added a bonus evaluation section to compare fuzzy text similarity between groundth truth and predicted in the ./notebooks/Qwen2.5.ipynb notebook.
4. Finally I have tried only LORA based finetuning. To improve accuracy I would have tried full finetuning, full finetuning+freezing vision layer etc. This is left for future work.


## Containerization
1. Leveraged fastAPI for model serving restAPI endpoint.


### To run the docker 
```bash
brew install colima # alternative to docker desktop
colima start --cpu 4 --memory 8
docker run -d -p 8000:8000 -e HF_TOKEN=<YOUR_HF_TOKEN> --memory="6g" --name qwen-vl-api qwen-vl-api
python infer.py # for inference
# for http://localhost:8000/docs for swagger docs
```

### To run directly
```bash
conda create --name qwen_infer python=3.11
conda activate qwen_infer
pip install -r requirements.txt
python app.py &
python infer.py
```
### Small note
Some files have the username as `suman muralidhar`, the laptop belongs to my wife and I forgot to change the username.