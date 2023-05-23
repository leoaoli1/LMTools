import torch
import scipy
import gradio as gr
from langchain import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from diffusers import DiffusionPipeline, AudioLDMPipeline


class SDImage:
    def __init__(self):
        print("Initializing Text-to-Image Generator")
        self.img_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5") 
        self.img_pipeline.to("cuda")

    def inference(self, prompt):
        out_image = self.img_pipeline(prompt).images[0]
        return out_image


class AudioLDM:
    def __init__(self):
        print("Initializing Text-to-Audio Generator")
        self.audio_pipeline = AudioLDMPipeline.from_pretrained("cvssp/audioldm", torch_dtype=torch.float16)
        self.audio_pipeline.to("cuda")

    def inference(self, prompt):
        out_audio = self.audio_pipeline(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
        return out_audio


class CoordinatorBot:
    def __init__(self):
        self.llm = OpenAI(
            temperature=0,
            model_name="text-davinci-003"
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.sdimage = SDImage()
        self.audioldm = AudioLDM()

    def init_tools(self):
        audio_tool = Tool(
            name='Text-to-Audio',
            func=self.audioldm.inference,
            description='Use this tool to generate an audio from text'
        )

        img_tool = Tool(
            name='Text-to-Image',
            func=self.sdimage.inference,
            description='Use this tool to generate an image from text'
        )


        tools = [audio_tool, img_tool]

        self.agent = initialize_agent(
            agent="conversational-react-description", 
            tools=tools,
            llm=self.llm,
            verbose=True,
            return_intermediate_steps=True, 
            memroy = self.memory
        )
        return None

    def run_text(self, message, chat_history):
        img = None
        audio = None
        res = self.agent({"chat_history": chat_history,"input": message})
        chat_history.append(message, res['output'])
        for action in res['intermediate_steps']:
            if action[0].tool == "Text-to-Image":
                print('Update image output')
                img = gr.Image.update(action[1])
                action[1].save('img.jpg')
                chat_history = chat_history + [(('img.jpg',), None)]
            elif action[0].tool == "Text-to-Audio":
                print('Update audio output')
                audio = gr.Audio.update((16000, action[1]))
                scipy.io.wavfile.write('audio.wav', rate=16000, data=action[1])
                chat_history = chat_history + [(('audio.wav',), None)]
        return "", chat_history, img, audio
    
    def clear_button(self):
        return gr.Button.update(visible=False)
    

if __name__ == '__main__':
    bot = CoordinatorBot()
    bot.init_tools()
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Markdown("## LMTools")
        chatbot = gr.Chatbot(elem_id="chatbot", label="LMTools", visible=True).style(height=500)

        state = gr.State([])

        with gr.Row() as text_input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            with gr.Column(scale=0.3, min_width=0):
                clear = gr.Button("Clear️")
        
        with gr.Row() as outputs:
            with gr.Column(scale=0.6):
                img = gr.Image(interactive=False)
            with gr.Column(scale=0.4):
                audio = gr.Audio(interactive=False)

        txt.submit(bot.run_text, [txt, chatbot], [txt, chatbot, img, audio])
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda:None, None, txt)

    demo.launch(share=True, server_name="127.0.0.1", server_port=7001)