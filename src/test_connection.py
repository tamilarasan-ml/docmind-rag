import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Read config
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Create client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version
)

# Send a simple test message
response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "user", "content": "Say hello in one short sentence and confirm you're working."}
    ]
)

print("SUCCESS! Model responded:")
print(response.choices[0].message.content)