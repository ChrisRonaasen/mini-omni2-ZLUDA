import sys
# ------------------- ZLUDA ---------------------------
import os
import torch
# ------------------- HIDE ROCM HIP -------------------
os.environ.pop("ROCM_HOME", None)
os.environ.pop("HIP_HOME", None)
os.environ.pop("ROCM_VERSION", None)

paths = os.environ["PATH"].split(";")
paths_no_rocm = [p for p in paths if "rocm" not in p.lower()]
os.environ["PATH"] = ";".join(paths_no_rocm)

# Fix for cublasLt errors on newer ZLUDA (if no hipblaslt)
os.environ['DISABLE_ADDMM_CUDA_LT'] = '1'

zluda_device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
is_zluda = zluda_device_name.endswith("[ZLUDA]")

# ------------------- BEGIN ZLUDA Audio Patch ---------
# Copied from Patientx' ComfyUI-Zluda rep, but adapted to fix the issues with 'window' and 'input' arguments
if is_zluda:
    _torch_stft = torch.stft
    _torch_istft = torch.istft

    def z_stft(input: torch.Tensor, *args, **kwargs):
        # Extract the 'window' argument from kwargs if it exists
        window = kwargs.pop('window', None)
        if window is not None:
            window = window.cpu()
            # Pass 'input' positionally and avoid passing it again as a keyword argument
            return _torch_stft(input.cpu(), *args, window=window, **kwargs).to(input.device)

    def z_istft(input: torch.Tensor, window: torch.Tensor, *args, **kwargs):
        return _torch_istft(input=input.cpu(), window=window.cpu(), *args, **kwargs).to(input.device)

    def z_jit(f, *_, **__):
        f.graph = torch._C.Graph()
        return f

    torch._dynamo.config.suppress_errors = True
    torch.stft = z_stft
    torch.istft = z_istft
    torch.jit.script = z_jit
# ------------------- END ZLUDA Audio Patch -----------

# ------------------- ZLUDA Top-K Fallback Patch ------
# no idea if this is necessary, copied willy-nilly from Patientx' ComfyUI-Zluda
if is_zluda:
    _topk = torch.topk

    def safe_topk(input: torch.Tensor, *args, **kwargs):
        device = input.device
        values, indices = _topk(input.cpu(), *args, **kwargs)
        return torch.return_types.topk((values.to(device), indices.to(device),))

    torch.topk = safe_topk
# ------------------- ZLUDA End Top-K Patch -----------

# ------------------- ONNX Runtime Patch --------------
# no idea if this is too necessary, but requirements.txt has onnxruntime in it, so i just, as always, copied willy-nilly from Patientx' ComfyUI-Zluda
try:
    import onnxruntime as ort

    if is_zluda:
        print("\n***----------------------ZLUDA-----------------------------***")
        print("  ::  Patching ONNX Runtime for ZLUDA — disabling CUDA EP.")

        # Store original get_available_providers
        original_get_available_providers = ort.get_available_providers

        def filtered_providers():
            return [ep for ep in original_get_available_providers() if ep != "CUDAExecutionProvider"]

        # Patch ONLY the _pybind_state version (used during session creation)
        ort.capi._pybind_state.get_available_providers = filtered_providers

        # Wrap InferenceSession to force CPU provider when CUDA is explicitly requested
        OriginalSession = ort.InferenceSession

        class SafeInferenceSession(OriginalSession):
            def __init__(self, *args, providers=None, **kwargs):
                if providers and "CUDAExecutionProvider" in providers:
                    print("  ::  Forcing ONNX to use CPUExecutionProvider instead of CUDA.")
                    providers = ["CPUExecutionProvider"]
                super().__init__(*args, providers=providers, **kwargs)

        ort.InferenceSession = SafeInferenceSession
except ImportError:
    print("  ::  ONNX Runtime not installed — skipping patch.")
except Exception as e:
    print("  ::  Failed to patch ONNX Runtime:", e)
# ------------------- End ONNX Patch ------------------

# ------------------- ZLUDA Backend -------------------
# no idea HOW MUCH of this is necessary, copied willy-nilly from Patientx' ComfyUI-Zluda, and I'm assuming the checks for functions is what made it work out of the box
if is_zluda:
    print("  ::  ZLUDA detected, disabling non-supported functions.      ")
    torch.backends.cudnn.enabled = False

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    print("  ::  CuDNN, flash_sdp, mem_efficient_sdp disabled.           ")
 
if is_zluda:
    print(f"  ::  Using ZLUDA with device: {zluda_device_name}")
    print("***--------------------------------------------------------***\n")
else:
    print(f"  ::  CUDA device detected: {zluda_device_name or 'None'}")
    print("***--------------------------------------------------------***\n")
# ------------------- End ZLUDA Backend ---------------


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import flask
import base64
import tempfile
import traceback
from flask import Flask, Response, stream_with_context
from inference_vision import OmniVisionInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0') -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False

        self.client = OmniVisionInference(ckpt_dir, device)
        self.client.warm_up()

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def chat(self) -> Response:

        req_data = flask.request.get_json()
        try:
            audio_data_buf = req_data["audio"].encode("utf-8")
            audio_data_buf = base64.b64decode(audio_data_buf)
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            image_data_buf = req_data.get("image", None)
            if image_data_buf:
                image_data_buf = image_data_buf.encode("utf-8")
                image_data_buf = base64.b64decode(image_data_buf)

            audio_path, img_path = None, None
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_f, \
                 tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                audio_f.write(audio_data_buf)
                audio_path = audio_f.name

                if image_data_buf:
                    img_f.write(image_data_buf)
                    img_path = img_f.name
                else:
                    img_path = None

                if img_path is not None:
                    resp_generator = self.client.run_vision_AA_batch_stream(audio_f.name, img_f.name,
                                                                             stream_stride, max_tokens,
                                                                             save_path='./vision_qa_out_cache.wav')
                else:
                    resp_generator = self.client.run_AT_batch_stream(audio_f.name, stream_stride,
                                                                      max_tokens,
                                                                      save_path='./audio_qa_out_cache.wav')
                return Response(stream_with_context(self.generator(resp_generator)),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
        except Exception as e:
            print(traceback.format_exc())
            return Response("An error occurred", status=500)

    def generator(self, resp_generator):
        for audio_stream, text_stream in resp_generator:
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: audio/wav\r\n\r\n'
            yield audio_stream
            yield b'\r\n--frame\r\n'
            yield b'Content-Type: text/plain\r\n\r\n'
            yield text_stream.encode()


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)
    return server.server

#not sure if it was necessary to use this port instead, but it's what streamlit kept using when opened, and things are working now, so fuck it! We doin' it live!
def serve(ip='0.0.0.0', port=8501, device='cuda:0'):

    OmniChatServer(ip, port=port,run_app=True, device=device)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)

