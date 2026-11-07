import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

print([m.name for m in client.models.list() if "argon" in m.name.lower()])

r = client.models.generate_content(model="gemini-4-argon", contents="Hello")
print(r.text)