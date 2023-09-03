import gradio as gr
from joblib import load
import os
import pandas as pd

# load the model
def load_model(path=""):
    if os.path.exists(path):
        model_dict = load(path)
        return model_dict
    else:
        print("Model not found")

model_dict = load_model('diamond_price.joblib')
# prediction function
def diamond_price_regressor(caret, depth, table, x, y, z, cut, color, clarity):
    model = model_dict['model']
    target_converter = model_dict['quantile']
    input_frame = pd.DataFrame({
        'carat': [caret],
        'cut': [cut],
        'color': [color],
        'clarity': [clarity],
        'depth': [depth],
        'table': [table],
        'x': [x],
        'y': [y],
        'z': [z]
    })
    print(input_frame)
    pred = model.predict(input_frame)
    pred = target_converter.inverse_transform(pred.reshape(-1, 1))
    print(pred)
    return f'Approx price is ${pred[0][0]:.2f}'

cut_choices = ['Ideal', 'Premium', 'Good', 'Very Good', 'Fair']
color_choices = ['E', 'I', 'J', 'H', 'F', 'G', 'D']
clarity_choices = ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']

# gradio interface
ui = gr.Interface(
    fn = diamond_price_regressor,
    inputs = [
        gr.Slider(minimum=0, maximum=10, step=.01, value=.7, label="Carat", info="1 carat = 0.2 grams"),
        gr.Slider(minimum=0, maximum=100, step=.01, value=61, label="Depth", info="Total depth percentage"),
        gr.Slider(minimum=0, maximum=100, step=.01, value=57, label="Table", info="Width of top of diamond relative to widest point"),
        gr.Slider(minimum=0, maximum=100, step=.01, value=5, label="x", info="Length in mm"),
        gr.Slider(minimum=0, maximum=100, step=.01, value=5, label="y", info="Width in mm"),
        gr.Slider(minimum=0, maximum=100, step=.01, value=3.5, label="z", info="Height in mm"),
        gr.Dropdown(cut_choices, label="Cut", value="Ideal", info="Cut quality"),
        gr.Dropdown(color_choices, label="Color", value="E", info="Color grade"),
        gr.Dropdown(clarity_choices, label="Clarity", value="SI2", info="Clarity grade")
    ],
    outputs = "text",
)

if __name__ == "__main__":
    ui.launch()