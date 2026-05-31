import os
# Memaksa OpenMP dan ONNX Runtime untuk menggunakan lebih banyak thread CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import streamlit as st
import cv2
import tempfile
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

st.title("EduReflect: Analisis Emosi Kelas")

# 1. Widget Upload Video
uploaded_video = st.file_uploader("Upload Rekaman Pembelajaran (MP4/MOV)", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    # Simpan ke file sementara agar OpenCV bisa baca
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    st.video(uploaded_video) # Tampilkan video pratinjau
    
    if st.button("Mulai Analisis"):
        # 2. Inisialisasi KEDUA Model & Konfigurasi Video
        face_model = YOLO('src/yolov11n-face.onnx') 
        emotion_model = YOLO('src/best_int8_openvino_model') 
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # --- Ambil FPS dari video untuk perhitungan waktu ---
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: # Validasi jika metadata FPS kosong/NaN
            fps = 30.0 
        
        # Inisialisasi hitungan emosi
        emotion_counts = {
            "Happy": 0, "Neutral": 0, "Angry": 0, "Contempt": 0,
            "Sad": 0, "Surprised": 0, "Fear": 0, "Disgust": 0
        }
        total_detections = 0
        timeline_data = []
        
        # Set untuk menyimpan ID wajah unik
        unique_face_ids = set()
        
        # List untuk menampung frame hasil deteksi (maksimal 60 frame agar hemat RAM)
        saved_frames_pool = []
        
        # Penanda Progress di Streamlit
        status_text = st.empty()
        status_text.write("⏳ Sedang melacak wajah dan menganalisis emosi siswa... Mohon tunggu.")
        progress_bar = st.progress(0)
        
        frame_idx = 0
        
        # --- KONFIGURASI OPTIMASI CPU ---
        FRAME_SKIP = 5  # Analisis 1 dari setiap 5 frame
        TARGET_WIDTH = 640 # Mengecilkan resolusi frame
        
        # 3. PROSES LOOPING VIDEO (TWO-STAGE DETECTION + TRACKING)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            
            # --- OPTIMASI 1: FRAME SKIPPING ---
            if frame_idx % FRAME_SKIP != 0:
                continue 
            
            # --- OPTIMASI 2: RESIZE FRAME ---
            h, w = frame.shape[:2]
            aspect_ratio = h / w
            target_height = int(TARGET_WIDTH * aspect_ratio)
            frame = cv2.resize(frame, (TARGET_WIDTH, target_height))
            
            annotated_frame = frame.copy()
            face_detected_in_this_frame = False
            time_in_seconds = frame_idx / fps
            
            # --- TAHAP 1: Deteksi & Lacak Wajah ---
            face_results = face_model.track(frame, conf=0.4, persist=True, verbose=False)
            
            for r_face in face_results:
                if r_face.boxes is not None and r_face.boxes.id is not None:
                    boxes = r_face.boxes.xyxy
                    track_ids = r_face.boxes.id.int().tolist()
                    
                    for box, track_id in zip(boxes, track_ids):
                        unique_face_ids.add(track_id)
                        
                        # Ambil koordinat kotak pembatas dan cegah keluar batas frame
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(TARGET_WIDTH, x2), min(target_height, y2)
                        
                        # Potong area wajah dari frame (Crop)
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Label default jika model emosi kurang yakin (< 0.40)
                        label_name = "Unknown" 
                        
                        if face_crop.size > 0:
                            # --- TAHAP 2: Klasifikasi Emosi ---
                            emotion_results = emotion_model(face_crop, conf=0.4, verbose=False)
                            
                            for r_emotion in emotion_results:
                                if len(r_emotion.boxes) > 0:
                                    top_box = r_emotion.boxes[0]
                                    cls_id = int(top_box.cls)
                                    label_name = emotion_model.names[cls_id].capitalize()
                                    
                                    if label_name in emotion_counts:
                                        emotion_counts[label_name] += 1
                                        total_detections += 1
                                        face_detected_in_this_frame = True
                                        timeline_data.append({
                                            "Waktu (detik)": time_in_seconds,
                                            "Emosi": label_name
                                        })
                        
                        # Gambar kotak wajah proporsional dan teks emosi
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"ID:{track_id} {label_name}", (x1, y1 - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Simpan frame ke pool jika ada emosi yang terdeteksi
            if face_detected_in_this_frame and len(saved_frames_pool) < 60:
                # Hitung dan format waktu (MM:SS)
                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)
                timestamp_text = f"{minutes:02d}:{seconds:02d}"
                
                # Gambar kotak hitam kecil dan teks waktu di pojok kiri atas
                cv2.rectangle(annotated_frame, (10, 10), (100, 40), (0, 0, 0), -1)
                cv2.putText(annotated_frame, timestamp_text, (15, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Konversi ke RGB untuk Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                saved_frames_pool.append(rgb_frame)
                                
        cap.release()
        
        # Selesai Proses Scan
        status_text.empty()
        progress_bar.empty()
        st.success("🎉 Analisis Selesai!")
        st.write("---")
        
        # --- 4. TAMPILKAN RINGKASAN DATA ---
        st.subheader("Ringkasan Hasil Analisis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Wajah/Siswa Terdeteksi", value=f"{len(unique_face_ids)} Orang")
        with col2:
            st.metric(label="Total Deteksi Emosi (Sampel Terpilih)", value=f"{total_detections} Kali")
            
        st.write("---")
        
        # --- 5. TAMPILKAN 4 KEY FRAMES HASIL ANALISIS ---
        if len(saved_frames_pool) > 0:
            st.subheader("Cuplikan Rekaman Analisis (Key Frames)")
            img_cols = st.columns(4)
            
            # Ambil 4 gambar secara merata dari awal hingga akhir video
            pool_size = len(saved_frames_pool)
            if pool_size >= 4:
                indices = [0, pool_size // 3, (pool_size * 2) // 3, pool_size - 1]
                selected_frames = [saved_frames_pool[i] for i in indices]
            else:
                selected_frames = saved_frames_pool 
                
            for idx, img_frame in enumerate(selected_frames):
                with img_cols[idx]:
                    st.image(img_frame, caption=f"Cuplikan {idx+1}", use_container_width=True)
            st.write("---")
        
        # --- 6. HITUNG PERSENTASE & PLOTLY ---
        if total_detections > 0:
            labels = []
            percentages = []
            counts = []
            
            for emotion, count in emotion_counts.items():
                pct = (count / total_detections) * 100
                labels.append(emotion)
                percentages.append(round(pct, 1))
                counts.append(count)
                
            df = pd.DataFrame({
                "Kategori Emosi": labels,
                "Persentase (%)": percentages,
                "Jumlah Terdeteksi (Wajah x Frame)": counts
            })
            
            fig = px.bar(
                df, 
                x="Kategori Emosi", 
                y="Persentase (%)", 
                text="Persentase (%)",
                color="Kategori Emosi",
                title="<b>Mood Breakdown (Profil Emosi Kelas)</b>",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hover_data=["Jumlah Terdeteksi (Wajah x Frame)"]
            )
            
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(showlegend=False, yaxis_range=[0, 110])
            
            st.subheader("Grafik Analisis")
            st.plotly_chart(fig, use_container_width=True)
            st.write("---")

            # --- 7. GRAFIK TREN EMOSI BERDASARKAN WAKTU ---
            if len(timeline_data) > 0:
                st.subheader("Tren Emosi Berdasarkan Waktu")

                df_timeline = pd.DataFrame(timeline_data)

                # Tentukan ukuran bin otomatis berdasarkan durasi video
                max_time = df_timeline["Waktu (detik)"].max()
                if max_time <= 120:
                    bin_size = 10   # Video pendek: bin 10 detik
                elif max_time <= 600:
                    bin_size = 30   # Video sedang: bin 30 detik
                else:
                    bin_size = 60   # Video panjang: bin 1 menit

                df_timeline["bin"] = (df_timeline["Waktu (detik)"] // bin_size) * bin_size
                df_grouped = (
                    df_timeline
                    .groupby(["bin", "Emosi"])
                    .size()
                    .reset_index(name="Jumlah Deteksi")
                    .sort_values("bin")
                )

                # Format label sumbu X: detik → MM:SS
                df_grouped["Waktu"] = df_grouped["bin"].apply(
                    lambda s: f"{int(s // 60):02d}:{int(s % 60):02d}"
                )

                fig_trend = px.line(
                    df_grouped,
                    x="Waktu",
                    y="Jumlah Deteksi",
                    color="Emosi",
                    markers=True,
                    title=f"<b>Tren Emosi Kelas per {bin_size} Detik</b>",
                    labels={"Jumlah Deteksi": "Jumlah Deteksi", "Waktu": "Waktu (MM:SS)"},
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_trend.update_layout(
                    xaxis_tickangle=-45,
                    legend_title_text="Emosi",
                )
                st.plotly_chart(fig_trend, use_container_width=True)
                st.write("---")
            
            # --- 8. REKOMENDASI BERBASIS EMOSI DOMINAN ---
            st.subheader("Rekomendasi Pengajaran")
            st.caption(
                "Rekomendasi disusun berdasarkan **Pekrun's Control-Value Theory of Achievement Emotions (2006)** "
                "dan prinsip *affective computing* dalam konteks pembelajaran. "
                "Gunakan sebagai bahan refleksi, bukan penilaian tunggal."
            )

            # Peta tiap emosi → (interpretasi, saran, tipe pesan)
            EMOTION_GUIDE = {
                "Happy": (
                    "Positive Activating Emotion",
                    "Suasana kelas kondusif dan siswa antusias. "
                    "Pertahankan ritme dan metode pengajaran saat ini. "
                    "Manfaatkan momentum ini untuk memperkenalkan materi yang lebih menantang.",
                    "info"
                ),
                "Neutral": (
                    "Ambiguous State (Focused OR Disengaged)",
                    "Neutral bisa berarti konsentrasi penuh (flow state) atau kebosanan pasif. "
                    "Lakukan pengecekan pemahaman (quick poll/pertanyaan lisan) untuk "
                    "memastikan siswa benar-benar mengikuti, bukan sekadar diam.",
                    "info"
                ),
                "Surprised": (
                    "Positive/Negative Activating Emotion",
                    "Kejutan bisa menandakan momen 'aha' (positif) atau kebingungan mendadak (negatif). "
                    "Perhatikan konteks: apakah muncul saat materi baru diperkenalkan? "
                    "Jika ya, manfaatkan sebagai jembatan diskusi.",
                    "info"
                ),
                "Sad": (
                    "Negative Deactivating Emotion",
                    "Emosi ini mengindikasikan rendahnya motivasi atau rasa tidak mampu. "
                    "Berikan penguatan positif (positive reinforcement), kecilkan target sementara, "
                    "dan pastikan siswa merasa aman untuk bertanya.",
                    "warning"
                ),
                "Angry": (
                    "Negative Activating Emotion (Frustration)",
                    "Umumnya muncul akibat frustrasi terhadap materi yang terlalu sulit atau "
                    "merasa tidak diperlakukan adil. Evaluasi kembali tingkat kesulitan soal/materi "
                    "dan beri ruang bagi siswa untuk mengekspresikan kesulitannya.",
                    "warning"
                ),
                "Fear": (
                    "Negative Activating Emotion (Anxiety)",
                    "Kecemasan akademik dapat secara langsung menghambat proses kognitif. "
                    "Kurangi tekanan evaluasi, normalkan kesalahan sebagai bagian dari belajar, "
                    "dan pertimbangkan aktivitas low-stakes sebelum penilaian utama.",
                    "warning"
                ),
                "Disgust": (
                    "Strong Negative Emotion",
                    "Emosi kuat yang bisa menandakan siswa merasa konten tidak relevan atau "
                    "pendekatan pengajaran kurang sesuai. Tinjau kembali relevansi materi "
                    "dengan konteks kehidupan siswa.",
                    "warning"
                ),
                "Contempt": (
                    "Strong Negative Emotion (Disengagement)",
                    "Sinyal disengagement yang serius. Pertimbangkan pendekatan yang lebih "
                    "student-centered, libatkan siswa dalam menentukan arah diskusi, "
                    "atau cari tahu hambatan non-akademis yang mungkin ada.",
                    "warning"
                ),
            }

            max_emotion = df.loc[df['Persentase (%)'].idxmax()]['Kategori Emosi']
            max_pct = df['Persentase (%)'].max()

            guide = EMOTION_GUIDE.get(max_emotion)

            if guide:
                interpretation, suggestion, msg_type = guide
                message = (
                    f"**Emosi Dominan:** **{max_emotion} ({max_pct}%)** "
                    f"— dikategorikan sebagai *{interpretation}*\n\n"
                    f"**Saran:** {suggestion}\n\n"
                    f"**Catatan:** Data dari ±{len(unique_face_ids)} wajah unik yang tertangkap kamera."
                )
                if msg_type == "info":
                    st.info(message)
                else:
                    st.warning(message)
        else:
            st.error("❌ Tidak ada wajah siswa yang terdeteksi di dalam video. Pastikan kualitas video cukup jelas.")