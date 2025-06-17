from enum import Enum
from typing import Any, Optional, Callable
from dmi.data.coco import COCOLoader
from dmi.data.base import BaseLoader
from dmi.data.openvid import OpenvidLoader
from dmi.data.audiocaps import AudioCapsLoader
from dmi.data.sharegpt4v import ShareGPT4VLoader
from dmi.data.clothodetail import ClothoDetailLoader
from dmi.data.sharegpt4video import ShareGPT4VideoLoader
from dmi.data.chebi20 import CHEBI20Loader
from dmi.data.candels import CANDELSLoader
from dmi.data.sydney import SydneyLoader
from transformers import ProcessorMixin

class Modality(str, Enum):
    IMAGE = 'image'
    AUDIO = 'audio'
    VIDEO = 'video'
    TEXT = 'text'
    MOLECULE = 'molecule'
    SATELLITE = 'satellite'
    GALAXY = 'galaxy'


DATA_CLASSES: dict[str, BaseLoader]  = {
    'coco': COCOLoader,
    'audiocaps': AudioCapsLoader,
    'openvid': OpenvidLoader,
    'sharegpt4v': ShareGPT4VLoader,
    'clothodetail': ClothoDetailLoader,
    'sharegpt4video': ShareGPT4VideoLoader,
    'chebi20': CHEBI20Loader,
    'candels': CANDELSLoader,
    'sydney': SydneyLoader,
}

DATA_MODALITIES: dict[str, Modality]  = {
    'coco': Modality.IMAGE,
    'audiocaps': Modality.AUDIO,
    'openvid': Modality.VIDEO,
    'sharegpt4v': Modality.IMAGE,
    'clothodetail': Modality.AUDIO,
    'sharegpt4video': Modality.VIDEO,
    'chebi20': Modality.MOLECULE,
    'candels': Modality.GALAXY,
    'sydney': Modality.SATELLITE,
}

MODEL_MODALITIES: dict[str, Modality] = {
    'openai/clip-vit-large-patch14': Modality.IMAGE,
    'timm/caformer_b36.sail_in22k': Modality.IMAGE,
    'clap-htsat-fused': Modality.AUDIO,
    'alibaba-pai/VideoCLIP-XL': Modality.VIDEO,
    'timm/ViT-L-16-SigLIP2-384': Modality.IMAGE,
    'Cacophony': Modality.AUDIO,
    'OpenGVLab/ViCLIP-B-16': Modality.VIDEO,
    'chendelong/RemoteCLIP-RN50-Unchanged': Modality.SATELLITE,
    'chendelong/RemoteCLIP-ViT-B-32-Unchanged': Modality.SATELLITE,
    'chendelong/RemoteCLIP-ViT-L-14': Modality.SATELLITE,
    'acharkq/MolCA': Modality.MOLECULE,
    'mwalmsley/zoobot-encoder-convnext_base': Modality.GALAXY,
    'mwalmsley/zoobot-encoder-convnext_tiny': Modality.GALAXY,
    'mwalmsley/zoobot-encoder-convnext_nano': Modality.GALAXY,
}

MODEL_CLASSES: dict[str, Any] = {
    'openai/clip-vit-large-patch14': None,
    'timm/caformer_b36.sail_in22k': None,
    'clap-htsat-fused': None,
    'MCG-NJU/videomae-base': None,
    'MCG-NJU/videomae-base-finetuned-kinetics': None,
    'alibaba-pai/VideoCLIP-XL': None,
    'timm/ViT-L-16-SigLIP2-384': None,
    'Cacophony': None,
    'OpenGVLab/ViCLIP-B-16': None,
    'chendelong/RemoteCLIP-RN50-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-B-32-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-L-14': None,
    'acharkq/MolCA': None,
    'mwalmsley/zoobot-encoder-convnext_base': None,
    'mwalmsley/zoobot-encoder-convnext_tiny': None,
    'mwalmsley/zoobot-encoder-convnext_nano': None,
}

PROCESSOR_CLASSES: dict[str, ProcessorMixin] = {
    'openai/clip-vit-large-patch14': None,
    'clap-htsat-fused': None,
    'alibaba-pai/VideoCLIP-XL': None,
    'timm/ViT-L-16-SigLIP2-384': None,
    'Cacophony': None,
    'OpenGVLab/ViCLIP-B-16': None,
    'chendelong/RemoteCLIP-RN50-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-B-32-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-L-14': None,
    'acharkq/MolCA': None,
    'mwalmsley/zoobot-encoder-convnext_base': None,
    'mwalmsley/zoobot-encoder-convnext_tiny': None,
    'mwalmsley/zoobot-encoder-convnext_nano': None,
}

F_POST_PROCESSORS: dict[str, Callable] = {
    'openai/clip-vit-large-patch14': None,
    'clap-htsat-fused': None,
    'alibaba-pai/VideoCLIP-XL': None,
    'timm/ViT-L-16-SigLIP2-384': None,
    'Cacophony': None,
    'OpenGVLab/ViCLIP-B-16': None,
    'chendelong/RemoteCLIP-RN50-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-B-32-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-L-14': None,
    'acharkq/MolCA': None,
    'mwalmsley/zoobot-encoder-convnext_base': None,
    'mwalmsley/zoobot-encoder-convnext_tiny': None,
    'mwalmsley/zoobot-encoder-convnext_nano': None,
}

EMBEDDING_NAMES: dict[str, Optional[str]] = {
    'openai/clip-vit-large-patch14': None,
    'clap-htsat-fused': None,
    'alibaba-pai/VideoCLIP-XL': None,
    'timm/ViT-L-16-SigLIP2-384': None,
    'Cacophony': None,
    'OpenGVLab/ViCLIP-B-16': None,
    'chendelong/RemoteCLIP-RN50-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-B-32-Unchanged': None,
    'chendelong/RemoteCLIP-ViT-L-14': None,
    'acharkq/MolCA': None,
    'mwalmsley/zoobot-encoder-convnext_base': None,
    'mwalmsley/zoobot-encoder-convnext_tiny': None,
    'mwalmsley/zoobot-encoder-convnext_nano': None,
}

# Define the chat template with the {% generation %} tag
LLAMA31_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {%- if message.role != 'assistant' %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
        {%- elif message.role == 'assistant' %}
            {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}
            {% generation %}
            {{- message['content'] | trim + '<|eot_id|>' }}
            {% endgeneration %}
        {%- endif %}    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


LLAMA32_CHAT_TEMPLATE = """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
            {%- if message.role != 'assistant' %}
                  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
            {%- elif message.role == 'assistant' %}
                  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}
                 {% generation %}
                 {{- message['content'] | trim + '<|eot_id|>' }}
                 {% endgeneration %}
           {%- endif %}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

LLMS_CHATTEMPLATES = {
    'meta-llama/Llama-3.1-8B-Instruct': LLAMA31_CHAT_TEMPLATE,
    'meta-llama/Llama-3.1-70B-Instruct': LLAMA31_CHAT_TEMPLATE,
    'meta-llama/Llama-3.2-1B-Instruct': LLAMA32_CHAT_TEMPLATE,
    'meta-llama/Llama-3.2-3B-Instruct': LLAMA32_CHAT_TEMPLATE,
}