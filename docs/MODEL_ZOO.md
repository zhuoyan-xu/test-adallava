# Model Zoo


If you are interested in including any other details in Model Zoo, please open an issue :)

The usage of LLaVA checkpoints should comply with the base LLM's model license.

AdaLLaVA follows the LLaVA v1.5 architecture, with [CLIP-ViT-L-336px](https://huggingface.co/openai/clip-vit-large-patch14-336) as the visual encoder (336*336 image resolution), [Vicuna-v1.5-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5) or [Vicuna-v1.5-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5) as the base LLM and a two-layer MLP as the vision-language connector. The saved model checkpoints can be downloaded from the following Hugging Face Repository.


## Ada-LLaVA-L



| Version | Size | Schedule | Base LLM | Vision Encoder | Projection layer | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ada-LLaVA | 7B | full_finetune-1_epoch | Vicuna-v1.5-7B | CLIP-ViT-L-336px | MLP-2x | [zhuoyanxu/ada-llava-v1.5-7b](https://huggingface.co/zhuoyanxu/ada-llava-v1.5-7b) |  
| Ada-LLaVA | 13B | full_finetune-1_epoch | Vicuna-v1.5-13B | CLIP-ViT-L-336px | MLP-2x | [zhuoyanxu/ada-llava-v1.5-13b](https://huggingface.co/zhuoyanxu/ada-llava-v1.5-13b) |  


**Note:** The above Ada-LLaVA checkpoints are saved from the original LLaVA repository, which is not directly compatible with the Transformers, i.e., it can not be directly loaded by ```LlavaForConditionalGeneration.from_pretrained('zhuoyanxu/ada-llava-v1.5-13b')```.
