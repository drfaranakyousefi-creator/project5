import torch
import json
from server_net import prediction_net


class Transmitter:
    def __init__(self, d_in: int, device: torch.device):
        self.device = device
        self.model  = prediction_net(
            d_in         = d_in,
            n_input_caps = 4,
            n_output_caps= 3,
            in_caps_dim  = 6,
            out_caps_dim = 8,
            n_routing    = 3,
            lr           = 0.001,
        ).to(device)

    def _to_json(self, x, label, status):
        return json.dumps({
            'input' : x.detach().cpu().tolist(),
            # باگ قبلی: label همیشه list بود حتی در test
            'label' : label.detach().cpu().tolist() if status == 'train' else [],
            'status': status,
        })

    def send_data(self, x, label, status):
        # ─── شبیه‌سازی ارسال به server ───
        payload = json.loads(self._to_json(x, label, status))

        result  = self.model(
            payload['input'],
            payload['label'] if status == 'train' else None,
            payload['status'],
        )

        # ─── شبیه‌سازی دریافت از server ───
        response = json.loads(json.dumps(result))

        if status == 'train':
            return torch.tensor(response['grad'], dtype=torch.float32).to(self.device)
        else:
            # بهبود: مستقیم tensor برمیگردونه با shape (batch,)
            return torch.tensor(
                response['prediction'], dtype=torch.float32
            ).to(self.device).squeeze(-1)