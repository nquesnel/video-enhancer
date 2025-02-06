import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from tqdm import tqdm

class VideoEnhancer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()

    def _load_model(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model_path = load_file_from_url(model_url, model_dir='weights')
        loadnet = torch.load(model_path, map_location=self.device)
        model.load_state_dict(loadnet['params_ema'], strict=True)
        model.eval()
        model = model.to(self.device)
        return model

    def enhance_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*4, height*4))

        with torch.no_grad():
            for _ in tqdm(range(total_frames), desc='Enhancing video'):
                ret, frame = cap.read()
                if not ret:
                    break

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).float().div(255.0)
                img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)

                output = self.model(img)
                output = output.squeeze().permute(1, 2, 0).cpu().numpy()
                output = (output * 255.0).round().astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                out.write(output)

        cap.release()
        out.release()
        cv2.destroyAllWindows()