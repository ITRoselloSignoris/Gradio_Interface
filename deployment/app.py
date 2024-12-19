import pandas as pd
import pickle
import gradio as gr

PARAMS_NAME=[
    "Age",
    "Class",
    "Wifi",
    "Booking",
    "Seat",
    "Checkin"
]

COLUMNS_PATH = "ohe_categories.pkl"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)

with open("rf.pkl", "rb") as f:
    model = pickle.load(f)


def predict_passenger_satisfaction(*args):
    answer_dict = {}
    
    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict) 

    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns=ohe_tr).fillna(0)

    prediction = model.predict(single_instance_ohe)

    class_map = {1:"Yes", 0:"No"}

    score = prediction.map(class_map)

    response = {"Passenger Satisfaction: ":score}

    return response

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Satisfacción aerolínea:airplane::earth_americas::airplane:
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
            """
            ## ¿Cliente satisfecho?
            """
            )
            Age_input = gr.Slider(label="Edad",minimum=7,maximum=85,step=1,randomize = True)
            Class_input = gr.Radio(
                label="Clase",
                choices=["Eco","EcoPlus","Business"],
                multiselect=False,
                value= "Eco"
            )
            Wifi_input = gr.Slider(label="Servicio de Wifi",minimum=0,maximum=5,step=1,randomize = True)
            Booking_input = gr.Slider(label="Facilidad de registro",minimum=0,maximum=5,step=1,randomize = True)
            Seat_input = gr.Dropdown(
                label="Comodidad del asiento",
                choices=[0,1,2,3,4,5],
                multiselect=False,
                value=0,
            )
            Checkin_input = gr.Dropdown(
                label="Experiencia con el Checkin",
                choices=[0,1,2,3,4,5],
                multiselect=False,
                value=0,
            )

        with gr.Column():
            """
            ## Predicción
            """
            target = gr.Label(label="Score")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict_passenger_satisfaction,
                inputs=[
                    Age_input,
                    Class_input,
                    Wifi_input,
                    Booking_input,
                    Seat_input,
                    Checkin_input,
                ],
                outputs=[target],
            )

    gr.Markdown(
        """
        <p style="align-center:center">
            <a href="https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science"
                target="_blank">Proyecto demo creado en el Bootcamp de Edvai 
            </a>
        </p>
        """
    )

demo.launch()