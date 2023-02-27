import openai 
import numpy as np
import pandas as pd
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv('unicorns_with_embeddings.csv')

def get_embedding(text):
  # Note how this function assumes you already set your Open AI key!
    result = openai.Embedding.create(
      model='text-embedding-ada-002',
      input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(vec1,vec2):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(vec1), np.array(vec2))

def embed_prompt_lookup():
    # initial question
    question = input("What question do you have about a Unicorn company? ")
    # Get embedding
    prompt_embedding = get_embedding(question)
    # Get prompt similarity with embeddings
    # Note how this will overwrite the prompt similarity column each time!
    df["prompt_similarity"] = df['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))

    # get most similar summary
    summary = df.nlargest(1,'prompt_similarity').iloc[0]['summary'] 

    prompt = f"""Only answer the question below if you have 100% certainty of the facts, use the context below to answer.
            Here is some context:
            {summary}
            Q: {question}
            A:"""


    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=500,
        model="text-davinci-003"
    )
    print(response["choices"][0]["text"].strip(" \n"))


if __name__ == "__main__":
    embed_prompt_lookup()