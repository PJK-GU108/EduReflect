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
        face_model = YOLO('yolov11n-face.pt') 
        emotion_model = YOLO('best.pt') 
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Inisialisasi hitungan emosi
        emotion_counts = {
            "Happy": 0, "Neutral": 0, "Angry": 0, 
            "Sad": 0, "Surprised": 0, "Fear": 0, "Disgust": 0
        }
        total_detections = 0
        
        # --- FITUR BARU: Set untuk menyimpan ID wajah unik ---
        unique_face_ids = set()
        
        # Penanda Progress di Streamlit
        status_text = st.empty()
        status_text.write("⏳ Sedang melacak wajah dan menganalisis emosi siswa... Mohon tunggu.")
        progress_bar = st.progress(0)
        
        frame_idx = 0
        
        # 3. PROSES LOOPING VIDEO (TWO-STAGE DETECTION + TRACKING)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            progress_bar.progress(frame_idx / total_frames)
            
            # --- TAHAP 1: Deteksi & Lacak Wajah dengan YOLOv11-Face Tracking ---
            # Menggunakan .track() dan persist=True agar ID wajah tetap konsisten antar frame
            face_results = face_model.track(frame, conf=0.4, persist=True, verbose=False)
            
            for r_face in face_results:
                # Pastikan ada objek dan ID yang terdeteksi
                if r_face.boxes is not None and r_face.boxes.id is not None:
                    boxes = r_face.boxes.xyxy
                    track_ids = r_face.boxes.id.int().tolist()
                    
                    for box, track_id in zip(boxes, track_ids):
                        # Simpan ID wajah ke dalam set (duplikat otomatis diabaikan)
                        unique_face_ids.add(track_id)
                        
                        # Ambil koordinat kotak pembatas
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Potong area wajah dari frame (Crop)
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # Validasi apakah hasil crop valid
                        if face_crop.size > 0:
                            # --- TAHAP 2: Klasifikasi Emosi ---
                            emotion_results = emotion_model(face_crop, conf=0.4, verbose=False)
                            
                            for r_emotion in emotion_results:
                                if len(r_emotion.boxes) > 0:
                                    top_box = r_emotion.boxes[0] # Ambil deteksi teratas
                                    cls_id = int(top_box.cls)
                                    label_name = emotion_model.names[cls_id].capitalize()
                                    
                                    if label_name in emotion_counts:
                                        emotion_counts[label_name] += 1
                                        total_detections += 1
                        
        cap.release()
        
        # Selesai Proses Scan
        status_text.empty()
        progress_bar.empty()
        st.success("🎉 Analisis Selesai!")
        st.write("---")
        
        # --- 4. TAMPILKAN RINGKASAN DATA (TERMASUK TOTAL WAJAH) ---
        st.subheader("Ringkasan Hasil Analisis")
        
        # Menggunakan kolom metrik Streamlit agar terlihat profesional
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Wajah/Siswa Terdeteksi", value=f"{len(unique_face_ids)} Orang")
        with col2:
            st.metric(label="Total Deteksi Emosi (Frame akumulatif)", value=f"{total_detections} Kali")
            
        st.write("---")
        
        # 5. HITUNG PERSENTASE & SIAPKAN DATA
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
            
            # 6. BUAT BAR CHART PLOTLY
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
            
            # Tampilkan Grafik
            st.subheader("Grafik Analisis")
            st.plotly_chart(fig, use_container_width=True)
            st.write("---")
            
            # 7. KESIMPULAN OTOMATIS
            st.subheader("Rekomendasi Pengajaran")
            
            max_emotion = df.loc[df['Persentase (%)'].idxmax()]['Kategori Emosi']
            max_pct = df['Persentase (%)'].max()
            
            if max_emotion in ["Happy", "Neutral"]:
                st.info(
                    f"👉 **Insight Utama:** Kelas didominasi oleh emosi **{max_emotion} ({max_pct}%)**, "
                    f"dari total sekitar {len(unique_face_ids)} siswa yang aktif bergerak/tertangkap kamera. "
                    "Menandakan suasana belajar kondusif.\n\n"
                    "💡 **Saran Perbaikan:** Pertahankan ritme mengajar Anda. Sesi interaktif sudah berjalan efektif."
                )
            else:
                st.warning(
                    f"👉 **Insight Utama:** Terdeteksi tingkat emosi **{max_emotion} sebesar {max_pct}%** di dalam kelas.\n\n"
                    f"💡 **Saran Perbaikan:** Angka emosi negatif ({max_emotion}) yang cukup tinggi menandakan siswa mengalami kendala. "
                    "Disarankan untuk mengevaluasi kembali bagian materi yang rumit, memberikan jeda *ice breaking*, atau "
                    "memperlambat tempo penjelasan pada pertemuan berikutnya."
                )
        else:
            st.error("❌ Tidak ada wajah siswa yang terdeteksi di dalam video. Pastikan kualitas video cukup jelas.")
