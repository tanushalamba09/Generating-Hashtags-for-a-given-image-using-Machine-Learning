# Generating-Hashtags-for-a-given-image-using-Machine-Learning
#the code is as follows
import os
import requests
from PIL import Image
from io import BytesIO
from transformers import pipeline, set_seed
from transformers import TFAutoModelForCausalLM, GPT2Config
config = GPT2Config.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = TFAutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", from_pt=True, config=config)
def generate_hashtags_for_image(file_path, num_hashtags=5):
    set_seed(42)
    image = Image.open(file_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    generator = pipeline('text-generation', model=model)
    text = generator(image, max_length=50, num_return_sequences=num_hashtags, do_sample=True)
    hashtags = []
    for i in range(num_hashtags):
        hashtag = '#' + text[i]['generated_text'].split('#')[1]
        hashtags.append(hashtag)  
    return hashtags
file_path = 'C:\\Users\\tanus\\Downloads\\Screenshot 2023-02-03 at 3.56.22 PM.png'
hashtags = generate_hashtags_for_image(file_path)
print(hashtags)
