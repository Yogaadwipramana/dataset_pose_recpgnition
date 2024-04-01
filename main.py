import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import mediapipe as mp

# Muat model yang telah dilatih
model_path = 'pose_recognition_model_Vdua.pth'
model = models.resnet18(pretrained=False)  # Menggunakan model ResNet-18
num_ftrs = model.fc.in_features
num_actions = 4
model.fc = nn.Linear(num_ftrs, num_actions)
model.load_state_dict(torch.load(model_path))
model.eval()

# Pilih perangkat (CPU atau GPU jika tersedia)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Pemetaan aksi
action_mapping = {
    0: "Perancangan Baut",
    1: "Pembukaan Cetakan",
    2: "pembersihan Cetakan",
    3: "Pelumasan",
}

# Transformasi untuk pengujian
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Flag untuk melacak apakah pose valid terdeteksi
valid_pose_detected = False

# Inisialisasi variabel untuk menyimpan pose referensi
reference_pose = None

# Inisialisasi variabel untuk menyimpan pose yang diprediksi
predicted_pose = None

# Tentukan jumlah frame yang harus konsisten sebelum mengenali aktivitas
consistent_frame_count = 5
current_frame_count = 0

while True:
    # Tangkap frame dari webcam
    ret, frame = cap.read()

    # Ubah frame menjadi RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Gunakan MediaPipe Pose untuk menemukan landmark
    results = pose.process(rgb_frame)

    # Gambar skeleton pada frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Jika pose valid terdeteksi, perbarui flag dan tampilkan pose yang diprediksi
        if not valid_pose_detected:
            valid_pose_detected = True
            # Simpan pose referensi saat pertama kali terdeteksi
            if reference_pose is None:
                reference_pose = action_mapping.get(predicted_pose, "Aksi Unknown")
    else:
        # Jika tidak ada pose yang terdeteksi, reset flag dan tampilkan "Aksi Tidak Dikenal"
        valid_pose_detected = False

        # Reset jumlah frame saat pose tidak dikenal
        current_frame_count = 0

    # Ubah frame untuk PyTorch
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image_tensor = test_transform(pil_image).unsqueeze(0).to(device)

    # Prediksi pose
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_pose = predicted.item()

    # Dapatkan aksi yang sesuai dari pemetaan
    action = action_mapping.get(predicted_pose, "Unknown")

    # Periksa apakah pose yang diprediksi konsisten selama beberapa frame
    if valid_pose_detected and action != reference_pose:
        current_frame_count += 1
    else:
        current_frame_count = 0

    # Jika pose telah konsisten selama beberapa frame, kenali aktivitasnya
    if current_frame_count >= consistent_frame_count:
        print(f"Deteksi Aktivitas: {action}")
        # Lakukan logika pengenalan aktivitas di sini jika diperlukan
        # ...

    # Tampilkan pose yang diprediksi dan aksi pada frame hanya jika pose valid terdeteksi
    if valid_pose_detected:
        # Tampilkan "Tidak Dikenal" sampai pose yang diprediksi cocok dengan pose referensi
        if action == reference_pose:
            cv2.putText(frame, 'Pose Diprediksi: Unknown', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'Pose Diprediksi: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # else:
    #     cv2.putText(frame, 'Pose Diprediksi: Unknown', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('Pengenalan Pose', frame)

    # Hentikan loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan webcam dan tutup jendela tampilan
cap.release()
cv2.destroyAllWindows()
