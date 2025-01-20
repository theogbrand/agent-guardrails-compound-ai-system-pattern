# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
import json
import gradio as gr
from groq import Groq

# these expect to find a .env file at the directory above the lesson.                                                         # the format for that file is (without the comment)                                                                           # API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
# def load_env():
#     _ = load_dotenv(find_dotenv())

def load_world(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# def get_together_api_key():
#      load_env()
#      together_api_key = os.getenv("TOGETHER_API_KEY")
#      return together_api_key

def get_groq_api_key():
    groq_api_key = os.getenv("GROQ_API_KEY")
    return groq_api_key

def test_groq_connection():
    client = Groq(api_key=get_groq_api_key())
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello world who are you"}],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    
    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")

def get_game_state(inventory={}):
    world = load_world('ogb_world.json')
    kingdom = world['kingdoms']['Valtoria']
    town = kingdom['towns']["Brindlemark"]
    character = town['npcs']['Kaelin Darkhaven']
    # start = world['start']
    
    game_state = {
        "world": world['description'],
        "kingdom": kingdom['description'],
        "town": town['description'],
        "character": character['description'],
        # "start": start,
        "inventory": inventory
    }
    return game_state

def run_action(message, history, game_state):
    
    if(message == 'start game'):
        return game_state['start']
        
    system_prompt = """You are an AI Game master. Your job is to write what \
happens next in a player's adventure game.\
Instructions: \
You must on only write 1-3 sentences in response. \
Always write in second person present tense. \
Ex. (You look north and see...)"""
    
    world_info = f"""
World: {game_state['world']}
Kingdom: {game_state['kingdom']}
Town: {game_state['town']}
Your Character:  {game_state['character']}"""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": world_info}
    ]

    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})
           
    messages.append({"role": "user", "content": message})
    client = Groq(api_key=get_groq_api_key())
    model_output = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    
    result = model_output.choices[0].message.content
    return result

def start_game(main_loop, share=False):
    demo = gr.ChatInterface(
        main_loop,
        chatbot=gr.Chatbot(height=250, placeholder="Type 'start game' to begin"),
        textbox=gr.Textbox(placeholder="What do you do next?", container=False, scale=7),
        title="AI RPG",
        # description="Ask Yes Man any question",
        theme="soft",
        examples=["Look around", "Continue the story"],
        cache_examples=False,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
                           )
    demo.launch(share=share, server_name="0.0.0.0")


everyone_content_policy = {
    "policy": """O1: Violence and Hate.
Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics.
- Use slurs based on personal characteristics
- Write gruesome / graphic violence.
Can
- Write tame fantasy violence
O2: Sexual Content.
Should not
- Write any somewhat sexual content that mentions sexual functions or sexual body parts.
- Write any content that would be inappropriate for kids.
Can
- Write romantic content (including kissing, flirting etc...)
O3: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Write story depicting suicide.
O4: Profanity.
Should not
- Write with any profane language that wouldn't be appropriate for kids.
Can
- Write with kid appropriate language
O5: Regulated or Controlled Substances.
Should not
- Write stories that depicts, glorifies or encourages drug use."""
}

def is_safe(message):
    # Build the prompt with embedded values
    prompt = f"""[INST] Task: Check if there is unsafe content in 'user' messages in conversations according to our safety policy with the below categories.
    
    <BEGIN UNSAFE CONTENT CATEGORIES>
    {everyone_content_policy['policy']}
    <END UNSAFE CONTENT CATEGORIES>
    
    <BEGIN CONVERSATION>
    user: {message}
    
    Provide your safety assessment for user in the above conversation:
    - First line must read 'safe' or 'unsafe'.
    - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
    
    client = Groq(api_key=get_groq_api_key())

    response = client.chat.completions.create(
        model="llama-guard-3-8b",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    return result.strip() == 'safe'
