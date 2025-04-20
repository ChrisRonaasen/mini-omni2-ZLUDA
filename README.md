
# Mini-Omni2-ZLUDA

<p align="center">
    <img src="./data/figures/title_new.png" width="90%"/>
</p>


<p align="center">
💜 
<a href="https://github.com/patientx/ComfyUI-Zluda">Patientx-ComfyUI-Zluda</a>          |
<a href="https://github.com/lshqqytiger/ZLUDA">Ishqqytiger - ZLUDA</a>          |
<a href="https://github.com/brknsoul/ROCmLibs">brknsoul - ROCm Libraries</a>    |
<a href="https://github.com/vladmandic/sdnext/wiki/ZLUDA">Vladamantic - SD.NEXT</a>     |
💜</p>

<p align="center">
🤗 <a href="https://huggingface.co/gpt-omni/mini-omni2">Hugging Face</a>   | 📖 <a href="https://github.com/gpt-omni/mini-omni2">Github</a>|     📑 <a href="https://arxiv.org/abs/2410.11190">Technical report</a> 
</p>

<h5 align="center"> If you like little Omni2, please give them a star ⭐ and consider citing their <a href="(https://arxiv.org/abs/2410.11190)">paper</a>!</h2>

Hi!
I have no idea what I'm doing, and I'm a total newbie when it comes to programming, but I also happen to be pretty stubborn, so here's a ZLUDA fork of GPT-Mini-Omni2!

This fork works perfectly with Python 3.11 and Torch 2.6.0 on my machine, with the current setup.

For requirements.txt, all I changed was switching out each "==" with ">=" to ensure it doesn't downgrade my Torch version to 2.3.x, and I also uncommented the PyAudio==0.2.14 part, so you can skip the installation for that with streamlit. Makes things a bit simpler. I left the numpy commented, as I am assuming you already have a working ZLUDA setup, which likely means you have numpy, torch, all that jazz setup already.

To test this, because I'm on mobile data with about 300kb/s speed, I used the same torch libraries and setup etc as I use for ComfyUI-Zluda.

Because of this, MAKE SURE YOU HAVE A WORKING ZLUDA SETUP BEFORE DOING THIS! Python/Torch version shouldn't matter, so long as it's at least **at or above Torch 2.3.X, Python 3.10**.

When testing this, make sure you know to do CTRL + C in the terminal to abort your current operations. Again, I have no clue what I'm doing, and I do not take responsibility for the harm your actions could cause to your computer or your sanity.

### Step by step instructions:
* Have a working Torch + ZLUDA environment. Refer to Vladamantic's guide, brknsoul's repository for custom ROCm libraries, and Ishqqytiger's repositories for the latest ZLUDA release. I included 3.9.2 normal 64bit full version in this repo, which worked for me. I highly recommend using miniconda or a similar environment tool to set up your environments, and adding your ZLUDA folder to Path so you don't have to keep including a new one for a long command so you can instead just do zluda -- *somecommandhere*.

* Also get FFMPEG! Make sure you download it from their website and place it somewhere like C:/FFMPEG or similar, and make an environment variable to add C:\ffmpeg\bin to Path. *(tip, you can do refreshenv in a terminal to update environment variables for that terminal (NOT THE SYSTEM BTW, JUST THAT TERMINAL)*

    (**Also tip2, the tippening: just pressing the windows button and starting to type "environment" will likely give you a shortcut to the right window for this stuff, and then just click the bottom button saying "environment variables", double click "path" in the top box, click browse, then find the ffmpeg folder, go INSIDE the "bin" folder, and click OK, OK, OK, OK, then restart/refreshenv.**)

* Follow the guide below to install the requirements etc, while using this repo instead. I have gone ahead and changed the original readme to ensure it includes this repository instead, as well as changing the commands so that they work, like, at all. *Great software btw, mini-omni, thank you so much please dont hate me im defenseless :)*

### Here's me outlining the steps very simply for you *(parts in parentheses"()" are for tips/info, NOT to be included in the command itself. Also, please ignore the special colors, I have no idea how to avoid that, they don't mean anything here)*:

TERMINAL WINDOW 1:
```sh
conda create -n omni python=3.10        
(or 11, OR, just do ZLUDA etc for comfyUI, whatever your preference)

call venv/scripts/activate              
(if using your own python venv, might need to do the git clone and venv stuff first unless youre just re-using your ComfyUI environment, which is what I am doing myself.)

conda activate omni                     
(if using your own conda setup)

git clone https://github.com/ChrisRonaasen/mini-omni2-ZLUDA

cd mini-omni2-ZLUDA

pip install -r requirements.txt         
(if youre having issues here, --force-reinstall might help you, but I HIGHLY recommend making a backup of your venv/lib/site-packages folder first if so, as itll reinstall both the requirements AND their dependencies, which might break your setup, depending on what you already have installed.)

set API_URL=http://localhost:60808/chat
(this and the next command is one place where I was getting a lot of my issues, since I already knew what to do about ZLUDA. I have, HOPEFULLY, included a fixed version of this stuff, but i recommend doing this command in terminal window 2 as well, just in case.)

zluda -- python server.py --ip '0.0.0.0' --port 60808
(or zluda.exe. Change this to .\zluda\zluda.exe if you havent added the zluda folder to path, same way as ffmpeg)

*(Also, be sure youre using THIS repo. Otherwise, you WILL be getting issues at this step. If you keep task manager open to the GPU view, you should be seeing dedicated GPU memory usage go up at this step, and ZLUDA messages should be popping up. YOU SHOULD NOT BE SEEING ERRORS aside from a message talking about you using a CUDA device (heheh) that has Tensor Cores, then a "/torch.set_float32" message, then a couple concat messages, then NOTHING SCARY. Everything should be standing still after it successfully starts, and it should finally tell you that its running, and a message telling you how to quit the program when you want to. Nothing actually happens after this in this terminal, it is just to start the program in a server backend. Head to terminal window 2 after this.)*
```

TERMINAL WINDOW 2:
```sh
conda activate omni
(if using your own conda setup, otherwise do call /venv/scripts/activate)

set API_URL=http://localhost:60808/chat

zluda -- streamlit run webui/omni_streamlit.py
(or zluda.exe. Change this to .\zluda\zluda.exe if you havent added the zluda folder to path, same way as ffmpeg)
```
At this point, a window should automatically open in your browser. I recommend keeping both terminals visible so you can see any outputs. Terminal window 1 will start outputting stuff AFTER terminal window 2, which will constantly be giving you messages like duration_after_vad: 0.000 s, and the time there will change if it detects your voice talking. After that, everything should output correctly, you should automatically be hearing a TTS voice responding to what you said, and terminal window 1 will tell you in text what the AI is saying, kinda like subtitles!

*NOTE: I HAVEN'T TESTED THE VISION STUFF YET, SO DON'T ASSUME IT TO BE WORKING FLAWLESSLY!*


# ORIGINAL (BUT SOMEWHAT ADAPTED) README BEYOND THIS POINT

## Introduction
Mini-Omni2 is an **omni-interactive** model. It can **understand image, audio and text inputs and has end-to-end voice conversations with users**. Featuring **real-time voice output**, **omni-capable multimodal understanding** and flexible interaction **ability with interruption mechanism while speaking**.

<p align="center">
    <img src="./data/figures/framework.jpeg" width="100%"/>
</p>


## Updates

- **2024.10:** Release the model, technical report, inference and chat demo code.

## Features
✅ **Multimodal interaction**: with the ability to understand images, speech and text, just like GPT-4o.

✅ **Real-time speech-to-speech** conversational capabilities. No extra ASR or TTS models required, just like [Mini-Omni](https://github.com/gpt-omni/mini-omni).

<!-- ✅ **Streaming audio output**: with first-chunk latency of audio stream less than 0.3s. -->

<!-- ✅ **Duplex interaction**: hearing while speaking, it can be interrupted by key words like "stop omni". -->


## Demo

NOTE: need to unmute first.

https://github.com/user-attachments/assets/ad97ca7f-f8b4-40c3-a7e8-fa54b4edf155

## Install

Create a new conda environment and install the required packages (adapted note: change the python version to your preferred one, no need to add the part after 10 or 11, conda will automatically install the latest one for you):

```sh
conda create -n omni python=3.10
conda activate omni

git clone https://github.com/ChrisRonaasen/mini-omni2-ZLUDA
cd mini-omni2-ZLUDA
pip install -r requirements.txt
```
Keep in mind, if you're using a python venv, you'll want to do call venv/scripts/activate (while in the directory that has the venv folder of course) instead of the conda commands. Chances are this is the case if you strictly followed ZLUDA for sd.next or ComfyUI and are adapting that for this. Simply copying the venv folder over means you'll have to change venv/pyvenv.cfg to fit your setup, FYI. Found that out the hard way lol. This guide is written with the assumption you went with Conda, as is my fork of it, hence why there's nothing here aside from tidbits of text in the guide that actually helps you with that part, but google is helpful for anything related to python environments!
## Quick start

**Interactive demo**

- start server

NOTE: you need to start the server before running the streamlit or gradio demo with API_URL set to the server address.

(What the note above is trying to say is "use two terminals", one to do the python server.py command, and one to start the web UI. You can probably make a script to just run both or something, but again, I have no clue what I'm doing.) 

Skip the first step here if you're on windows, and instead just download ffmpeg from their website and add it to path. Then either restart, or do "refreshenv" in your terminal. You'll be using TWO terminals for this, so do refreshenv in both. Or restart. That's easier.
Also do "pip install ffmpeg" (without quotes). Just in case. No idea if necessary, but that's what I did first, which didn't help at all, but hey, maybe it will for you! I have no clue, but that's life sometimes! :)
```sh
sudo apt-get install ffmpeg
conda activate omni
cd mini-omni2-ZLUDA
python3 server.py --ip '0.0.0.0' --port 60808
```


- run streamlit demo

NOTE: you need to run streamlit **locally** with PyAudio installed.

```sh
pip install PyAudio==0.2.14
API_URL=http://0.0.0.0:60808/chat streamlit run webui/omni_streamlit.py
```


**Local test**

```sh
conda activate omni
cd mini-omni2
# test run the preset audio samples and questions
python inference_vision.py
```

## Mini-Omni2 Overview

**1. Multimodal Modeling**:
We use multiple sequences as the input and output of the model. In the input part, we will concatenate image, audio and text features to perform a series of comprehensive tasks, as shown in the following figures. In the output part, we use text-guided delayed parallel output to generate real-time speech responses.
<p align="center">
    <img src="./data/figures/inputids.png" width="100%"/>
</p>

**2. Multi-stage Training**:
We propose an efficient alignment training method and conduct encoder adaptation, modal alignment, and multimodal fine-tuning respectively in the three-stage training.
<p align="center">
    <img src="./data/figures/training.jpeg" width="100%"/>
</p>

<!-- **3. Cases**:
Here are more cases of Mini-Omni2:
<p align="center">
    <img src="./data/figures/samples.png" width="100%"/>
</p> -->

## FAQ

**1. Does the model support other languages?**

No, the model is only trained on English. However, as we use whisper as the audio encoder, the model can understand other languages which is supported by whisper (like chinese), but the output is only in English.

**2. Error: can not run streamlit in local browser, with remote streamlit server**
    
You need start streamlit **locally** with PyAudio installed.


## Acknowledgements 

- [Qwen2](https://github.com/QwenLM/Qwen2/) as the LLM backbone.
- [litGPT](https://github.com/Lightning-AI/litgpt/) for training and inference.
- [whisper](https://github.com/openai/whisper/)  for audio encoding.
- [clip](https://github.com/openai/CLIP)  for image encoding.
- [snac](https://github.com/hubertsiuzdak/snac/)  for audio decoding.
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for generating synthetic speech.
- [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [MOSS](https://github.com/OpenMOSS/MOSS/tree/main) for alignment.

## Citation

```bibtex
@article{xie2024miniomni2opensourcegpt4ovision,
      title={Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities}, 
      author={Zhifei Xie and Changqiao Wu},
      year={2024},
      eprint={2410.11190},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      journal={ArXiv},
      volume={abs/2410.11190},
}
```
## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=gpt-omni/mini-omni2&type=Date)](https://star-history.com/#gpt-omni/mini-omni2&Date)
