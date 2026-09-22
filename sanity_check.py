from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17ActionHead
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.groot_n1_7 import tree
import diffusers, transformers
assert tree is not None, 'dm-tree missing'
from transformers import Qwen3VLForConditionalGeneration
print('N1.7 OK | transformers', transformers.__version__, '| diffusers', diffusers.__version__)