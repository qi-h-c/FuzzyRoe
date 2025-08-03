#from .model import LlavaLlamaForCausalLM
import os, sys

# 计算项目根目录：train_mem.py 在  llava/train/ 下，向上两级就是项目根
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
