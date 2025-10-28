# 🚌 Bus Information Guidance System for the Visually Impaired

### 🌍 Project Overview
This project was developed to assist visually impaired individuals in recognizing approaching buses through computer vision and artificial intelligence technologies. By detecting motion, recognizing yellow license plates, and reading bus numbers in real-time, the system provides auditory feedback to help users identify incoming buses safely and efficiently.
The project’s primary goal is to enhance mobility independence for the visually impaired by reducing the uncertainty and risks associated with waiting for public transportation.

### 🎯 Objectives
- To detect buses approaching the camera using motion analysis and optical flow.
- To identify yellow license plates characteristic of Korean buses.
- To extract and recognize bus numbers through OCR-based image processing.
- To provide audio guidance that announces the bus number and direction, improving accessibility for visually impaired users.

### 🧠 Technical Summary
The system integrates multiple computer vision modules: <br>
1. Motion Detection: Frame differencing and histogram equalization are applied to distinguish moving objects in real time.
2. License Plate Recognition: HSV color thresholding isolates yellow regions corresponding to bus license plates.
3. Optical Flow Analysis: The system determines whether a detected bus is approaching or moving away using flow vector magnitude.
4. OCR (Optical Character Recognition): Tesseract OCR reads the bus number from the extracted plate region.
5. Auditory Output (TTS): The recognized information is converted into speech to inform users audibly. <br>

These components together form a fully automated detection pipeline capable of recognizing and announcing real-world bus information under varying lighting and motion conditions.

### ⚙️ Core Technologies
- Programming Language: Python
- Libraries: OpenCV, NumPy, Pytesseract, gTTS
- Approach: Frame-based motion detection, color segmentation, morphological processing, optical flow analysis, and OCR.

### 💡 Significance
This project demonstrates the potential of AI-assisted accessibility systems that combine computer vision with human-centered design. By automatically detecting buses and audibly announcing their numbers, it bridges the gap between digital technology and real-world inclusivity.
The solution not only showcases a technical achievement in motion and image analysis but also contributes meaningfully to social welfare and smart-city innovation by enabling safer, independent navigation for visually impaired passengers.
