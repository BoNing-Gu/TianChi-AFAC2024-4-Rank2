import time
import argparse
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description='Chatbot Interface with Customizable Parameters')
parser.add_argument('--model-url',
                    type=str,
                    default='http://localhost:8000/v1',
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
你好，你可以作为金融专家帮我分析文本吗？
"""
messages = [{"role": "user", "content": prompt}]
start = time.time()
response = client.chat.completions.create(
    model=args.model,
    messages=messages,
    stream=False,
    max_tokens=4096,
    temperature=args.temp
)
print(response.choices[0].message.content)