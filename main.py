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

    # a little strange but we are kind of "grouping" a patient's message to the AI as a message AND a state update,
    # then this run_action main agnet manages how to reply to the patient based on the previous instructions
    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})

    messages.append({"role": "user", "content": message})
    model_output = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview", messages=messages
    )

    result = model_output.choices[0].message.content
    return result

def update_patient_state(patient_state, state_updates):
    update_msg = ""
    if not state_updates:
        return update_msg, patient_state
    for update in state_updates:
        patient_state[update["patient_state_key"]] = update["patient_state_value"]
        update_msg += f"{update['patient_state_key']} updated to {update['patient_state_value']}\n"
    return update_msg, patient_state


def main_loop(message, history):
    # TODO: from collecting BP (1 step) to going through checklist (N steps)
    # Next TODO: Master Magenta One inner and outer loop with handover
    initial_patient_state = {
        "blood_pressure_taken": False,
        "blood_pressure_reading": None,
    }
    # TODO: Add MedPrompt+ here where each specialist agent goes through MedPrompt+ pipeline (scratchpad/not, dynamic few shot, randomised ensembling with majority vote)
    output = run_action(message, history, initial_patient_state)

    # TODO: Input Guard here, with callback for failing guard (Action, string message)
    # stage 1 guard: guard checks after complete LLM output (batch guard), API/DIY
    # stage 2 guard: guard checks after each LLM output token (streaming guard)
    # stage 3 guard: multiple parallel guards check after each LLM output token (batch guard)
    # stage 4 guard: multiple parallel guard checks after each LLM output token (streaming guard)
    # safe = is_safe(output)
    # if not safe:
    #     return 'Invalid Output'

    state_updates = detect_state_changes(initial_patient_state, output)
    # return "Patient State Updated with: " + str(state_updates)

    # TODO: after intent detection, transfer to specialist agent to execute workflow (think SWARM-type multi-agent framework)
    # TODO: some message-passing framework to pass state and messages to specialist agent, while constantly pruned (garbage collect assistant chat history to avoid confusion)
    update_msg, final_patient_state = update_patient_state(
        initial_patient_state, state_updates
    )

    # TODO: Output Guard here, **check model does not say anything offtopic, mention PII, mention other sensitive topics, only shows empathy and asks to take BP, otherwise return reask specialist agent with feedback [seems like this makes the most difference according to Hippo]
    # safe = is_safe(update_msg)
    # if not safe:
    #     return 'Invalid Output' # fail message should pass to global LLM state or some LLM fallback for cycle to recover

    # TODO: update state or checklist
    # update_msg = update_inventory(
    #     game_state['inventory'],
    #     item_updates
    # )
    output += update_msg

    return output


def start_chat(main_loop, share=False):
    # set variables for initial patient state here and pass it to main_loop
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
