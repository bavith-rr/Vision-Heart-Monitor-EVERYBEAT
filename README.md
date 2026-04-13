# 💓 Vision AI Heart Monitor (EveryBeat)

### 1. The Product Vision (The "Why")
Wearable health monitors (smartwatches, chest straps) are expensive and create a hardware barrier to entry for basic vitals tracking. Digital health platforms struggle to onboard users quickly because capturing baseline metrics usually requires manual data entry or external devices. 

**The Solution:** A frictionless web application that democratizes health monitoring. It utilizes a standard laptop/smartphone camera and deep learning to measure a user's heart rate (BPM) in real-time, eliminating the need for physical sensors.

### 2. Business Impact & Strategy
* **Frictionless Acquisition:** Acts as a high-conversion lead-magnet for telemedicine platforms by allowing users to test their vitals instantly without logging in.
* **Hardware-Agnostic Accessibility:** Drastically lowers the barrier to entry for digital health tools in emerging markets where expensive wearables are not prevalent.
* **Personalized UX:** Maps real-time AI predictions against the user's demographic data (age, weight, height) to deliver tailored medical insights rather than raw, confusing numbers.

---

### 3. Technical Architecture (The "How")
This application is built using a lightweight Flask MVC architecture, processing live video feeds through a custom Convolutional Neural Network (CNN).

**A. Computer Vision Pipeline (OpenCV)**
* The frontend captures real-time video via the browser `MediaDevices` API and transmits binary frames to the backend asynchronously.
* `cv2.imdecode` decodes the incoming byte stream, where frames are resized to a 64x64 resolution and normalized (`/ 255.0`) to match the exact input tensor shape required by the neural network.

**B. Deep Learning Engine (TensorFlow / Keras)**
* **Architecture:** A custom CNN model (`heartbeat_cnn_model.h5`) utilizing sequential `Conv2D` and `MaxPooling2D` layers for spatial feature extraction (detecting facial micro-color changes caused by blood flow).
* **Prediction Logic:** The model processes spatial features and models temporal sequences to output localized predictions, which the Python backend averages and scales to accurately calculate Beats Per Minute (BPM).

**C. Backend & State Management (Python / Flask)**
* **Stateful Routing:** Utilizes global dictionary mapping to temporarily hold user demographic data across the session.
* **Dynamic Medical Logic:** A conditional logic matrix evaluates the final BPM against established medical baselines (adjusted dynamically for age brackets), returning a customized JSON payload to the frontend.

### 4. Tech Stack Overview
* **AI / Deep Learning:** TensorFlow, Keras (CNN Architecture), NumPy
* **Computer Vision:** OpenCV (`cv2`)
* **Backend:** Python, Flask, Flask-CORS
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Canvas & MediaDevices API)
