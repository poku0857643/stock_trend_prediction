from openai import OpenAI
import tiktoken

def chunk_tokens(text, model="gpt-4", max_tokens = 500):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i: i+ max_tokens]
        chunks.append(encoding.decode(chunk))
        return chunks