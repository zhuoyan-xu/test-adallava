
# AdaLLaVA: Learning to Inference Adaptively for Multimodal Large Language Models


## TODO List
- [ ] Upload the model weights to Hugging Face
- [ ] The code for model training
- [ ] The code for model inference
- [ ] The code for evaluation
- [ ] Make the Table-LLaVA checkpoints compatible with the Transformers package (loadable via LlavaForConditionalGeneration.from_pretrained(''))


## Setup
```
conda create -n test-adallava python=3.10
conda activate test-adallava
cd src
bash install.sh
```


## Train

```
bash scripts/train_script.sh
```


## Acknowledgement


- [LLaVA](https://github.com/haotian-liu/LLaVA): The codebase we built upon.

- [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer): The code we used for calculating FLOPs and prefill time.

- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval): The code for evaluating multimodal LLMs.

