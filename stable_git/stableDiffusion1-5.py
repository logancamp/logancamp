from io import BytesIO
import os
import numpy as np
import requests
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionImageVariationPipeline
from tkinter import *
from PIL import Image, ImageTk
from csv import reader
import sv_ttk
from tkinter import filedialog
from skimage import io, transform

api_token = ''
os.environ["REPLICATE_API_TOKEN"] = api_token
# Now you can import the modules from the added path
import replicate

#set up window
gui = Tk()
gui.title("Stable Diffusion")
sv_ttk.set_theme("dark")

#make bark title bar
from dark_title_bar import*
dark_title_bar(gui)

#add general style selection psudo-globally
gen_styles = ["futuristic", 
            "realistic", 
            "horor", 
            "contemporary", 
            "anime", 
            "chartoon", 
            "minimalist", 
            "abstract", 
            "fantasy", 
            "digital art", 
            "sketch"]

models_list = ["runwayml/stable-diffusion-v1-5",  
               "nitrosocke/Ghibli-Diffusion", 
               "hakurei/waifu-diffusion", 
               "naclbit/trinart_stable_diffusion_v2", 
               "SG161222/Realistic_Vision_V1.4"
               "prompthero/openjourney",
               "lambdalabs/sd-pokemon-diffusers", 
               "CompVis/stable-diffusion-v1-4", 
               "stabilityai/stable-diffusion-2-1-base"]

#---------------------------------------------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------

class mainGui:
    def __init__(self, master):
        #set defaults
        self.NSFW = False
        self.model_id = None
        self.load_image = None

        #label for prompt entry
        self.pLabel = Label(gui, width=15, text="Enter prompt here:", font=("Verdana", 15), padx=10)
        self.pLabel.grid(row=0, column=0)
            
        #promt text entry box
        self.prompt = Entry(gui, width=75)
        self.prompt.grid(row=0, column=1, columnspan=2, padx=10, pady=8)

        #label for style entry
        self.sLabel = Label(gui, width=15, text="Enter styles here:", font=("Verdana", 15), padx=10)
        self.sLabel.grid(row=1, column=0)

        #style text entry box
        self.styles = Entry(gui, width=75)
        self.styles.grid(row=1, column=1, columnspan=2, padx=10, pady=8)

        #display general styles drop down
        self.gen_clicked = StringVar()
        self.gen_clicked.set("  add a style  ")
        self.gen_drop = OptionMenu(gui, self.gen_clicked, *gen_styles, command=self.addGenStyle)
        self.gen_drop.grid(row=0, column=3)

        #add image as input button
        self.img_but = Button(gui, height=3, text="Img\nto\nImg ", command=self.add_image_prompt)
        self.img_but.grid(row=0, column=5, rowspan=2, padx=5)
        #clear all inputs button
        self.clr_but = Button(gui, height=3, text="CLR \nTXT ", command=self.clear)
        self.clr_but.grid(row=0, column=6, rowspan=2, padx=5)
        #run button
        self.run_but = Button(gui, height=3, text="RUN", command=self.run)
        self.run_but.grid(row=0, column=7, rowspan=2, padx=5)

        #display model choice drop down
        self.model_clicked = StringVar()
        self.model_clicked.set("select model")
        self.model_drop = OptionMenu(gui, self.model_clicked, *models_list, command=self.choose_model)
        self.model_drop.grid(row=1, column=3)

#---------------------------------------------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------
    #choose model -- tip: all models must come from a huggingface repository
    def choose_model(self, event):
        model = str(self.model_clicked.get())
        self.model_id = model

    def add_image_prompt(self):
        self.load_image = self.upload_file()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png; *.gif")])
        image = io.imread(file_path)
        #mini display of prompt image
        self.inputlab = Label(gui, text="IMG:")
        self.inputlab.grid(row=6, column=7, pady=20)
        uplImage = transform.resize(image, (70, 70), anti_aliasing=True)
        uplImage = Image.fromarray((uplImage * 255).astype('uint8'))
        self.uplImage = ImageTk.PhotoImage(uplImage)
        self.inputBox = Label(gui, padx=10)
        self.inputBox.grid(row=6, column=8, pady=20)
        self.inputBox.configure(image=self.uplImage)
        #print and return
        print("Selected file:", file_path)
        return image

    #file save-path creation: for saving
    def uniqueFN(self, path):
        filename, extension = os.path.splitext(path)
        counter = 1

        while os.path.exists(path):
            path = filename + ' (' + str(counter) + ')' + extension
            counter += 1
        return path

    #save an image
    def saveImage(self, image, prompt):
        SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT')
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        image_path = self.uniqueFN(os.path.join(SAVE_PATH, (prompt[:25] + '...') if len(prompt)>25 else prompt) + '.png')
        image.save(image_path, format='PNG')
        return image_path

    #compile full prompt from prompt and styles
    def compilePromt(self, prompt, styles_ent):
        styles = self.compileStyles(styles_ent)
        if styles == [""]:
            compPrompt = prompt
        else:
            compPrompt = prompt + " in the style of " + (", and ").join(styles) + "."
        return compPrompt

#---------------------------------------------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------

    #get list of styles selected
    def compileStyles(self, styles):
        return list(styles.split(", "))

    #add all general styles to style list selection
    def addGenStyle(self, event):
        text = str(self.gen_clicked.get())
        if not self.styles.get():
            self.styles.insert(END, text)
        else:
            self.styles.insert(END, ", "+text)
        self.gen_clicked.set(gen_styles[0])
    
    #clear all text entries and styles
    def clear(self):
        self.prompt.delete(0,END)
        self.styles.delete(0,END)

#---------------------------------------------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------

    #initialize stable diffusion prior to image generation
    def stable_init_(self, type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if type == "standard":
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        elif type == "variations":
            pipe = StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers")
        elif type == "image":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        pipe = pipe.to(device)
        pipe.safety_checker = lambda images, clip_input: (images, self.NSFW)
        return (pipe, device)
    
    #generate image based on prompt
    def create_image(self, prompt, pipe, device):
        print('\n' + prompt)
        with autocast(device):
            image = pipe(prompt).images[0]
        return image
    
    #generate image from image and prompt
    def create_img_to_img(self, image, prompt, pipe, device):
        print(f"img_to_img: {prompt}")
        generator = torch.Generator(device=device).manual_seed(1024)
        with autocast(device):
            image = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
        return image

    #create a set number of variations from original and saves them to a folder
    def create_variations(self, num, image, prompt, pipe, device):
        SAVE_PATH = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'SD_OUTPUT', f'var-{prompt}')
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        with autocast(device):
            image = pipe(num*[image], guidance_scale=6.0).images

        image.save(image_path, format='PNG')
        for idx, im in enumerate(image):
            image_path = self.uniqueFN(os.path.join(SAVE_PATH, (prompt[:25] + '...') if len(prompt)>25 else prompt) + '.png')
            im.save(image_path, format='PNG')
            print(f"Saved variation {idx+1} at: {image_path}")

    def face_rep(self, image_path, prompt):
        output = replicate.run(
            "sczhou/codeformer:7de2ea26c616d5bf2245ad0d5e24f0ff9a6204578a5c876db53142edd9d2cd56",
            input={"image": open(image_path, "rb")})
        self.FPromLab = Label(gui, text=("Face Correction --" + (prompt[:60] + '...') if len(prompt)>60 else prompt), font=("Ariel", 10), padx=10)
        self.FPromLab.grid(row=3, column=1)
        output = requests.get(output, stream=True)
        new_output = Image.open(BytesIO(output.content))
        self.display_images(new_output, f"Face Correction -- {prompt}")
    
    #display image to gui
    def display_images(self, image, prompt):
        image = np.array(image)
        image = transform.resize(image, (512, 512), anti_aliasing=True)
        image = Image.fromarray((image * 255).astype('uint8'))
        self.dispImage = ImageTk.PhotoImage(image)
        self.imBox = Label(gui, padx=10)
        self.imBox.grid(row=4, column=0, columnspan=3, rowspan=5, pady=20)
        self.imBox.configure(image=self.dispImage)
        #save and skip button creation and display
        self.save_but = Button(text="Save Image", width=30, command=lambda: self.saveImage(image, prompt))
        self.save_but.grid(row=4, column=3, columnspan=2)
        self.skip_but = Button(text="Re-Run Prompt", width=30, command=self.run)
        self.skip_but.grid(row=5, column=3, columnspan=2)
        #img to img button creation and display
        self.itoi_but = Button(text="Re-Run with Image", width=30, command=lambda: self.run_im_to_im(image))
        self.itoi_but.grid(row=6, column=3, columnspan=2)
        #face repair button creation and display (will save original version only)
        self.itoi_but = Button(text="Save+Repair Face", width=30, command=lambda: self.face_rep(self.saveImage(image, prompt), prompt))
        self.itoi_but.grid(row=7, column=3, columnspan=2)
        #var num text entry box
        self.vars = Entry(gui, width=5)
        self.vars.grid(row=8, column=4)
        #variations button creation and display (will save original version and a folder with variations)
        self.var_but = Button(text="Save Variations:", width=20, command=lambda: self.run_variations(image))
        self.var_but.grid(row=8, column=3)
    
    #run stable diffusion functions on run button event
    def run(self):
        prom = str(self.prompt.get())
        style = str(self.styles.get())
        full_prompt = self.compilePromt(prom, style)
        if self.model_id is None or self.model_id == "select model":
            self.FPromLab = Label(gui, text="ERROR: Please select a valid model.", font=("Ariel", 10), padx=10)
            self.FPromLab.grid(row=3, column=1)
        else:
            self.FPromLab = Label(gui, text=((full_prompt[:70] + '...') if len(full_prompt)>70 else full_prompt), font=("Ariel", 10), padx=10)
            self.FPromLab.grid(row=3, column=1)
            if self.load_image is None:
                pipe = self.stable_init_("standard")
                image = self.create_image(full_prompt, *pipe)
            else:
                pipe = self.stable_init_("image")
                image = self.create_img_to_img(self.load_image, full_prompt, *pipe)
            self.display_images(image, full_prompt)
    
    #run stable diffusion functions on image to image button event
    def run_im_to_im(self, image):
        prom = str(self.prompt.get())
        style = str(self.styles.get())
        full_prompt = self.compilePromt(prom, style)
        self.FPromLab = Label(gui, text=((full_prompt[:70] + '...') if len(full_prompt)>70 else full_prompt), font=("Ariel", 10), padx=10)
        self.FPromLab.grid(row=3, column=1)
        pipe = self.stable_init_("image")
        image = self.create_img_to_img(image, full_prompt, *pipe)
        self.display_images(image, full_prompt)
        
    #run stable diffusion functions on variations button event
    def run_variations(self, image):
        prom = str(self.prompt.get())
        style = str(self.styles.get())
        full_prompt = self.compilePromt(prom, style)
        pipe = self.stable_init_("variations")
        image = self.create_variations(int(self.vars.get()), image, full_prompt, *pipe)

#initialize gui and begin the program
e = mainGui(gui)
gui.mainloop()   