import json
import gradio as gr


from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def detect_state_changes(patient_state, message_history):
    system_prompt = """You are an AI blood pressure monitoring assistant. \
    Your job is to detect changes to a patient's \
    blood pressure monitoring status based on the most recent chat history and patient state.
    If a patient mentions they took their blood pressure and provides a reading, update the patient's blood pressure reading with the provided reading and set the blood_pressure_taken to True.
    If a patient insists not to take their blood pressure, don't make any changes to the patient's blood pressure reading or blood_pressure_taken.
    Given a patient's name, patient state and chat history, return a list of json update
    of the patient's patient state in the following form.
    Only update the blood pressure reading and blood_pressure_taken if it is clear the patient took their blood pressure.
    If blood pressure was not taken return {"patientBloodPressureUpdates": []}
    and nothing else.

    Response must be in Valid JSON
    Don't alter any other patient state

    Patient Blood Pressure Updates:
    {
        "patientBloodPressureUpdates": [
            {"patient_state_key": <KEY>, 
            "patient_state_value": <VALUE>}...
        ]
    }
    """
    # TODO: use tool call with ENUM and strict_json schema validation here to prevent hallucinations and "determinism" in state update outcomes
    # for determinism in behavior of WHEN state changes are made, need evals, finetuning and real world testing

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Current Patient State: {str(patient_state)}",
        },  # game state (current inventory)
        {
            "role": "user",
            "content": f"Recent Messages: {str(message_history)}",
        },  # recent story
        {
            "role": "user",
            "content": "Patient Blood Pressure Updates",
        },  # request for state changes
    ]
    chat_completion = client.chat.completions.create(
        response_format={"type": "json_object"},
        # response_format={"type": "json_object", "schema": InventoryUpdate.model_json_schema()},
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=messages,
    )
    response = chat_completion.choices[0].message.content
    print("response from state change detector: ", response)
    # result = json.loads(response)
    try:
        result = json.loads(response)
        return result["patientBloodPressureUpdates"]
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        print("Error position:", e.pos)
        print("Error line:", e.lineno)
        print("Error column:", e.colno)


def run_action(message, history, patient_state):
    # if message == "start game":
    #     return patient_state["start"]

    system_prompt = """You are an AI Blood Pressure Monitoring assistant. Your job is to check if a patient has taken their blood pressure. If they have, update the patient's blood pressure reading and set the blood_pressure_taken to True. If they have not, empathise and ask them politely to take their blood pressure.
Instructions: \
1. Ask how the patient is doing today and if they have taken their blood pressure \
2. If the patient has not taken their blood pressure, ask them to take it \
3. If the patient has taken their blood pressure, ask them to provide their blood pressure reading \
"""

    patient_state_info = f"""
Patient State: 
blood_pressure_taken: {patient_state["blood_pressure_taken"]}
blood_pressure_reading: {patient_state["blood_pressure_reading"]}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": patient_state_info},
    ]

    # added by state change detector above so we can update the patient state later
    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})

    messages.append({"role": "user", "content": message})
    model_output = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=messages
    )

    result = model_output.choices[0].message.content
    return result


def main_loop(message, history):
    # change to fetch from DB
    patient_state = {
        "blood_pressure_taken": False,
        "blood_pressure_reading": 120,
    }
    output = run_action(message, history, patient_state)

    # safe = is_safe(output)
    # if not safe:
    #     return 'Invalid Output'

    state_updates = detect_state_changes(patient_state, output)
    return "Patient State Updated with: " + str(state_updates)

    # update_msg = update_inventory(
    #     game_state['inventory'],
    #     item_updates
    # )
    # output += update_msg

    # return output


def start_chat(main_loop, share=False):
    demo = gr.ChatInterface(
        main_loop,
        chatbot=gr.Chatbot(height=250, placeholder="Type 'start chat' to begin"),
        textbox=gr.Textbox(
            placeholder="What do you do next?", container=False, scale=7
        ),
        title="AI Blood Pressure Monitoring",
        theme="soft",
        examples=["my blood pressure is 120", "i haven't taken my blood pressure"],
        cache_examples=False,
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
    )
    demo.launch(share=share, server_name="0.0.0.0")


start_chat(main_loop, True)
