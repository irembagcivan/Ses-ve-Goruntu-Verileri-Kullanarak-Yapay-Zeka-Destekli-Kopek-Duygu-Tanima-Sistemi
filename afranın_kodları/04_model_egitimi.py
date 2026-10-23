import os
import copy
import ssl
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# macOS SSL sertifika doğrulama hatasını bypass eder:
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. DONANIM AYARI (Apple M2 GPU / MPS)
if torch.backends.mps.is_available():
    cihaz = torch.device("mps")
    print("Apple M2 GPU (MPS) aktif edildi!")
else:
    cihaz = torch.device("cpu")
    print("MPS bulunamadı, CPU kullanılıyor.")

# 2. DİZİN YOLLARI
KOD_DIZINI = Path(__file__).resolve().parent
PROJE_KOKU = KOD_DIZINI.parent
VERI_INDEKSI = PROJE_KOKU / "hazir_veri_seti" / "veri_seti_indeksi.csv"
VERI_KLASORU = PROJE_KOKU / "hazir_veri_seti"

df = pd.read_csv(VERI_INDEKSI)

# 3. VERİ DÖNÜŞÜMLERİ (Data Augmentation)
egitim_donusumu = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_donusumu = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. PYTORCH DATASET SINIFI
class KopekDuyguDataset(Dataset):
    def __init__(self, veri_df, kok_dizin, goruntu_donusum=None, spek_donusum=None):
        self.df = veri_df.reset_index(drop=True)
        self.kok_dizin = kok_dizin
        self.goruntu_donusum = goruntu_donusum
        self.spek_donusum = spek_donusum

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        satir = self.df.iloc[idx]
        
        goruntu_yolu = self.kok_dizin / satir["goruntu_dosyasi"]
        spek_yolu = self.kok_dizin / satir["spektrogram_dosyasi"]
        etiket = int(satir["sinif_id"])

        goruntu = Image.open(goruntu_yolu).convert("RGB")
        spektrogram = Image.open(spek_yolu).convert("RGB")

        if self.goruntu_donusum:
            goruntu = self.goruntu_donusum(goruntu)
        if self.spek_donusum:
            spektrogram = self.spek_donusum(spektrogram)

        return goruntu, spektrogram, etiket

train_df = df[df["veri_grubu"] == "train"]
test_df = df[df["veri_grubu"] == "test"]

train_dataset = KopekDuyguDataset(train_df, VERI_KLASORU, goruntu_donusum=egitim_donusumu, spek_donusum=egitim_donusumu)
test_dataset = KopekDuyguDataset(test_df, VERI_KLASORU, goruntu_donusum=test_donusumu, spek_donusum=test_donusumu)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. MODEL MİMARİLERİ
def tekli_resnet_olustur(sinif_sayisi=4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, sinif_sayisi)
    return model

class MultimodalModel(nn.Module):
    def __init__(self, sinif_sayisi=4):
        super().__init__()
        # Görüntü Omurgası
        resnet_gorsel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.gorsel_omurga = nn.Sequential(*list(resnet_gorsel.children())[:-1])
        
        # Ses Omurgası
        resnet_ses = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.ses_omurga = nn.Sequential(*list(resnet_ses.children())[:-1])
        
        # Birleştirme (Fusion) Katmanı: 512 + 512 = 1024
        self.siniflandirici = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, sinif_sayisi)
        )

    def forward(self, goruntu, spektrogram):
        f_gorsel = self.gorsel_omurga(goruntu).flatten(1)
        f_ses = self.ses_omurga(spektrogram).flatten(1)
        birlesik = torch.cat((f_gorsel, f_ses), dim=1)
        return self.siniflandirici(birlesik)

# 6. EĞİTİM FONKSİYONU
def model_egit(model, train_loader, test_loader, model_tipi="multimodal", epoch_sayisi=10):
    model = model.to(cihaz)
    kriter = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    en_iyi_dogruluk = 0.0
    en_iyi_agirliklar = copy.deepcopy(model.state_dict())

    for epoch in range(1, epoch_sayisi + 1):
        model.train()
        toplam_kayip = 0.0
        
        for goruntuler, spektrogramlar, etiketler in train_loader:
            goruntuler = goruntuler.to(cihaz)
            spektrogramlar = spektrogramlar.to(cihaz)
            etiketler = etiketler.to(cihaz)

            optimizer.zero_grad()
            if model_tipi == "gorsel":
                ciktilar = model(goruntuler)
            elif model_tipi == "ses":
                ciktilar = model(spektrogramlar)
            else:
                ciktilar = model(goruntuler, spektrogramlar)

            kayip = kriter(ciktilar, etiketler)
            kayip.backward()
            optimizer.step()
            toplam_kayip += kayip.item() * len(etiketler)

        # Doğrulama / Test Aşaması
        model.eval()
        tahminler, gercekler = [], []
        with torch.no_grad():
            for goruntuler, spektrogramlar, etiketler in test_loader:
                goruntuler = goruntuler.to(cihaz)
                spektrogramlar = spektrogramlar.to(cihaz)
                
                if model_tipi == "gorsel":
                    ciktilar = model(goruntuler)
                elif model_tipi == "ses":
                    ciktilar = model(spektrogramlar)
                else:
                    ciktilar = model(goruntuler, spektrogramlar)

                preds = torch.argmax(ciktilar, dim=1).cpu().numpy()
                tahminler.extend(preds)
                gercekler.extend(etiketler.numpy())

        dogruluk = accuracy_score(gercekler, tahminler)
        if dogruluk > en_iyi_dogruluk:
            en_iyi_dogruluk = dogruluk
            en_iyi_agirliklar = copy.deepcopy(model.state_dict())

        print(f"[{model_tipi.upper()}] Epoch {epoch}/{epoch_sayisi} - Kayıp: {toplam_kayip/len(train_dataset):.4f} - Test Doğruluk: %{dogruluk*100:.2f}")

    model.load_state_dict(en_iyi_agirliklar)
    f1 = f1_score(gercekler, tahminler, average="macro")
    return model, en_iyi_dogruluk, f1

# 7. ÇALIŞTIRMA VE KIYASLAMA
if __name__ == "__main__":
    print("\n--- 1. MODEL: SADECE GÖRSEL (ResNet18) ---")
    gorsel_model = tekli_resnet_olustur()
    _, gorsel_acc, gorsel_f1 = model_egit(gorsel_model, train_loader, test_loader, model_tipi="gorsel", epoch_sayisi=8)

    print("\n--- 2. MODEL: SADECE SES / SPEKTROGRAM (ResNet18) ---")
    ses_model = tekli_resnet_olustur()
    _, ses_acc, ses_f1 = model_egit(ses_model, train_loader, test_loader, model_tipi="ses", epoch_sayisi=8)

    print("\n--- 3. MODEL: MULTIMODAL (GÖRSEL + SES) ---")
    multi_model = MultimodalModel()
    en_iyi_multi, multi_acc, multi_f1 = model_egit(multi_model, train_loader, test_loader, model_tipi="multimodal", epoch_sayisi=8)

    print("\n================== NİHAİ BAŞARI TABLOSU ==================")
    sonuc_df = pd.DataFrame({
        "Model": ["Sadece Görsel", "Sadece Ses (Spektrogram)", "Multimodal (Görsel + Ses)"],
        "Test Doğruluk (Accuracy)": [f"%{gorsel_acc*100:.2f}", f"%{ses_acc*100:.2f}", f"%{multi_acc*100:.2f}"],
        "Macro F1-Score": [round(gorsel_f1, 4), round(ses_f1, 4), round(multi_f1, 4)]
    })
    print(sonuc_df.to_string(index=False))