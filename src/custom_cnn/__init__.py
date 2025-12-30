from main import *

if __name__=="__main__":
    demo = gr.Interface(fn=check_plant_disease, inputs=gr.Image(type="filepath"), outputs="text")
    demo.launch(debug=True)