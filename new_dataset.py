import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader


# ─── فیلتر کردن itemid های مربوطه ───────────────────────────────────────────
def filter_noisy_data(x, dataset_name):
    item_id = {
        'metavision': [
            220045,          # Heart Rate
            220210,          # Respiratory Rate
            220179, 220180,  # Non-invasive BP Mean
            220052,          # Arterial BP Mean
            220277           # SpO2 (هدف)
        ],
        'carevue': [
            211,             # Heart Rate
            618,             # Respiratory Rate
            52,              # Arterial BP Mean
            456,             # NBP Mean
            676, 678,        # Temperature
            646              # SpO2 (هدف)
        ]
    }
    return x[x['itemid'].isin(item_id[dataset_name])].copy()


# ─── استخراج داده از یک بیمار ────────────────────────────────────────────────
def extract_data_from_person(dataframe, W, dataset_name, target):
    N = 4 if dataset_name == 'metavision' else 5

    data, label, mask = [], [], []

    # باگ اصلی کد قبلی: e خارج از loop تعریف شده بود
    # و هیچوقت reset نمیشد — مقادیر قدیمی در sample های جدید باقی میموند
    e = torch.zeros(N)
    x = torch.zeros(W, N)
    m = torch.zeros(W)
    s = 0

    for _, row in dataframe.iterrows():
        item_id = row['itemid']
        value   = row['value']

        # رد کردن مقادیر نامعتبر
        try:
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
        except (ValueError, TypeError):
            continue

        # ─── تابع کمکی داخلی برای ذخیره sample و reset ───
        def save_and_reset():
            nonlocal s, x, m, e
            if s > 0:
                data.append(x.clone())
                mask.append(m.clone())
            x = torch.zeros(W, N)
            m = torch.zeros(W)
            e = torch.zeros(N)
            s = 0

        # ─── تابع کمکی برای ثبت یک مقدار در sequence ────
        def record(idx, val):
            nonlocal s, x, m, e
            if s < W:
                e[idx] = val
                x[s] = e.clone()   # مهم: clone بگیر نه reference
                m[s] = 1
                s += 1

        # ─── SpO2 ─────────────────────────────────────────
        if item_id in (646, 220277):
            if target == 'spO2':
                # SpO2 = label → sample قبلی رو ذخیره کن
                if s > 0:
                    data.append(x.clone())
                    label.append(value)
                    mask.append(m.clone())
                    x = torch.zeros(W, N)
                    m = torch.zeros(W)
                    e = torch.zeros(N)
                    s = 0
            elif target == 'BP':
                record(0, value)
            elif target == 'RR':
                record(1, value)

        # ─── Arterial BP ──────────────────────────────────
        elif item_id in (52, 220052):
            if target == 'BP':
                if s > 0:
                    data.append(x.clone())
                    label.append(value)
                    mask.append(m.clone())
                    x = torch.zeros(W, N)
                    m = torch.zeros(W)
                    e = torch.zeros(N)
                    s = 0
            else:
                record(0, value)

        # ─── Respiratory Rate ─────────────────────────────
        elif item_id in (618, 220210):
            if target == 'RR':
                if s > 0:
                    data.append(x.clone())
                    label.append(value)
                    mask.append(m.clone())
                    x = torch.zeros(W, N)
                    m = torch.zeros(W)
                    e = torch.zeros(N)
                    s = 0
            else:
                record(1, value)

        # ─── Heart Rate ───────────────────────────────────
        elif item_id in (211, 220045):
            record(2, value)

        # ─── Non-invasive BP (metavision) ─────────────────
        elif item_id in (220179, 220180):
            record(3, value)

        # ─── NBP Mean (carevue) ───────────────────────────
        elif item_id == 456:
            record(3, value)

        # ─── Temperature (carevue) ────────────────────────
        elif item_id in (676, 678):
            record(4, value)

    # تبدیل list به tensor
    if len(data) > 0:
        return (
            torch.stack(data, dim=0),
            torch.tensor(label, dtype=torch.float32),
            torch.stack(mask, dim=0),
        )
    # باگ کد قبلی: None برگردوندن برای tensor مشکل‌ساز بود
    return None, None, None


# ─── استخراج داده از کل dataset ─────────────────────────────────────────────
def extract_data(dataset_name, df_chartevents, w, target, normalize=True):
    all_data, all_labels, all_mask = [], [], []

    for subject_id in df_chartevents['subject_id'].unique():
        subject_df  = df_chartevents[df_chartevents['subject_id'] == subject_id]
        filtered_df = filter_noisy_data(subject_df, dataset_name)
        d, l, m     = extract_data_from_person(filtered_df, w, dataset_name, target)

        # باگ کد قبلی: if label != None  ← اشتباه برای tensor
        # درست: is not None
        if l is not None:
            all_data.append(d)
            all_labels.append(l)
            all_mask.append(m)

    if len(all_data) == 0:
        raise ValueError("هیچ داده‌ای استخراج نشد — dataset_name یا target رو چک کن")

    data   = torch.cat(all_data,   dim=0)
    labels = torch.cat(all_labels, dim=0)
    masks  = torch.cat(all_mask,   dim=0)

    if normalize:
        # بهبود: normalize per-feature به جای normalize کل داده یکجا
        # کد قبلی یک mean/std برای همه feature ها حساب میکرد که اشتباهه
        for i in range(data.shape[-1]):
            feat = data[:, :, i]
            mean = feat.mean()
            std  = feat.std()
            data[:, :, i] = (feat - mean) / (std + 1e-4)

    return data, labels, masks


# ─── Dataset class ───────────────────────────────────────────────────────────
class PatientDataset(Dataset):
    def __init__(self, data, label, mask):
        super().__init__()
        self.data  = data
        self.label = label
        self.mask  = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.mask[idx]


# ─── data_preparing ──────────────────────────────────────────────────────────
class data_preparing:
    def __init__(self, data_frame, dataset_name, w, test_size, target, batch_size):
        x, y, mask = extract_data(dataset_name, data_frame, w, target)

        n_train = int((1 - test_size) * x.shape[0])

        train_ds = PatientDataset(x[:n_train],  y[:n_train],  mask[:n_train])
        test_ds  = PatientDataset(x[n_train:],  y[n_train:],  mask[n_train:])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        self.test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)