import time
import argparse
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8001/v1',
                    help='Model URL')
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=True,
                    help='Model name for the chatbot')
parser.add_argument('--temp',
                    type=float,
                    default=0.1,
                    help='Temperature for text generation')
parser.add_argument('--stop-token-ids',
                    type=str,
                    default='',
                    help='Comma-separated stop token IDs')
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
prompt = """
请作为金融专家帮我分析文本吗。这句话是否存在逻辑词使用的错误？请给出你的推理逻辑。\n
上级审计机关对其审计管辖范围内的审计事项，可以授权下级审计机关进行审计，但本法第十八条至第二十条规定的审计事项不得进行授权；上级审计机关对下级审计机关审计管辖范围内的重大审计事项，可以直接进行审计，但是应当防止不必要的重复审计。
"""
messages = [{"role": "user", "content": prompt}]
start = time.time()
response = client.chat.completions.create(
    model=args.model,
    messages=messages,
    stream=False,
    max_tokens=1024,
    temperature=args.temp
)
print(response.choices[0].message.content)