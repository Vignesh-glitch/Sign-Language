import threading
import cv2
import mediapipe as mp
from tkinter import filedialog
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk, ImageSequence
import joblib
import whisper
import os
import numpy
import sounddevice as sd
import queue
import time
from scipy.io.wavfile import write
import difflib
import ttkbootstrap as tb
from ttkbootstrap import Window
from ttkbootstrap.constants import *
from googletrans import Translator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tkinter import filedialog
# Load gesture model and scaler
model_gesture = joblib.load(r"C:/Users/vigne/PycharmProjects/original project/gesture-recognition/gesture_model.pkl")
scaler = joblib.load(r"C:/Users/vigne/PycharmProjects/original project/gesture-recognition/gesture_scaler.pkl")
# Whisper model
whisper_model = whisper.load_model("base")
# TTS
engine = pyttsx3.init()
tts_lock = threading.Lock()
# Global variables
animation_id = None
running = True
frames = []
gif = None
whisper_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)
sample_rate = 16000
chunk_duration = 5

audio_queue = queue.Queue()
gesture_running = False
chart_visible = False


# Stats
usage_stats = {
    "voice_to_gesture": 0,
    "text_to_gesture": 0,
    "gesture_to_text_audio": 0
}
label_map = {
    "voice_to_gesture": "Voice-Gest",
    "text_to_gesture": "Text-Gest",
    "gesture_to_text_audio": "Gest-Text"
}
chart_canvas = None  # global chart canvas reference


# Tkinter root
root: Window = tb.Window(themename="flatly")
root.title("Sign Language Translator")
root.geometry("700x600")
# GUI panels
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=2)
# Language selector
language_options = ["en", "ta", "hi", "fr", "es"]
selected_language = tk.StringVar(value="en")
# Left panel for usage chart
left_panel = tb.Frame(root)
left_panel.grid(row=0, column=0, rowspan=1, sticky="nsew")
left_panel.grid_rowconfigure(1, weight=1)

# Center the grid widget in the window
root.grid_rowconfigure(1, weight=1)  # Center the first row
root.grid_columnconfigure(0, weight=1)  # Center the first column

# Initialize chart with empty/default figure
fig = Figure(figsize=(4, 3), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Feature Usage")
ax.set_ylabel("Count")
ax.set_xlabel("Feature")
bars = ax.bar([], [])

# Chart frame setup inside left panel
chart_frame = tb.Frame(left_panel)
chart_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Right panel for UI components
right_panel = tb.Frame(root)
right_panel.grid(row=0, column=1, rowspan=5, sticky="nsew", padx=10, pady=10)
right_panel.grid_columnconfigure(0, weight=1)

# Frame setup inside right panel
frame_1 = tb.LabelFrame(left_panel, text="Input Controls", padding=10)
frame_1.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
frame_1.grid_columnconfigure(0, weight=1)
frame_1.grid_columnconfigure(1, weight=1)

def speak_text(text):
    with tts_lock:
        engine.say(text)
        engine.runAndWait()
def log_event(text):
    root.after(0, lambda: (log.insert(tk.END, f"{text}\n"), log.see(tk.END)))
def change_language(*args):
    global selected_language
    selected_language = language_menu.get()

def translate_text(text):
    try:
        translator = Translator()
        return translator.translate(text, dest=selected_language.get()).text
    except Exception as e:
        log_event(f"Translation failed: {e}")
        return text


def load_custom_gif():
    file_path = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
    if file_path:
        word = os.path.splitext(os.path.basename(file_path))[0].lower()
        custom_dir = '../animations/'
        os.makedirs(custom_dir, exist_ok=True)
        new_path = os.path.join(custom_dir, f"{word}.gif")
        if not os.path.exists(new_path):
            with open(file_path, "rb") as src, open(new_path, "wb") as dst:
                dst.write(src.read())
            log_event(f"Loaded custom gesture: {word}")
        display_gesture_animation(word)

custom_gif_btn = tb.Button(frame_1, text="📥 Load Custom GIF", command=load_custom_gif, bootstyle=SECONDARY)
custom_gif_btn.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

frame_2 = tb.LabelFrame(left_panel, text="Gesture Animation", padding=10)
frame_2.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
translated_label = tb.Label(frame_2, text="", font=("Segoe UI", 14, "bold"))
translated_label.grid(row=0, column=0, pady=(0, 5))


# Keep the one inside frame_2:
gif_label = tb.Label(frame_2)
gif_label.grid(row=0, column=0, pady=10)
# Text input for the user
text_input_frame = tb.Frame(right_panel)
text_input_frame.grid(row=0, column=0, pady=(10, 0), padx=10, sticky="n")

# Create a text box (Entry widget)
text_input = tk.Entry(text_input_frame, width=40, font=("Segoe UI", 12))
text_input.grid(row=0, column=0, padx=10, pady=10)

def show_definition():
    word = text_input.get().strip().lower()
    if not word:
        log_event("No word entered.")
        return

    definition_window = tk.Toplevel(root)
    definition_window.title(f"Definition: {word}")
    definition_window.geometry("400x200")

    try:
        import wikipedia
        summary = wikipedia.summary(word, sentences=2)
    except Exception as e:
        summary = f"Definition not found.\n\n{e}"

    label = tk.Label(definition_window, text=summary, wraplength=380, justify="left")
    label.pack(padx=10, pady=10)


dict_btn = tb.Button(frame_2, text="🧑‍🏫 Show Definition", command=show_definition, bootstyle=INFO)
dict_btn.grid(row=2, column=0, pady=5)

frame_3 = tb.LabelFrame(left_panel, text="Status Log", padding=20)
frame_3.grid(row=2, column=0, padx=20, pady=10, sticky="ew")


text_input.grid(row=0, column=0, padx=10, pady=10)
text_input_frame.grid_columnconfigure(0, weight=1)
right_panel.grid_rowconfigure(0, weight=1)
right_panel.grid_columnconfigure(0, weight=1)

def update_translation():
    text = text_input.get()
    translated = translate_text(text)
    translated_label.config(text=f"Translated: {translated}")
    text_input.bind("<Return>", lambda e: (
        update_translation(),
        usage_stats.update({"text_to_gesture": usage_stats["text_to_gesture"] + 1}),
        display_gesture_animation(text_input.get().lower()),
        engine.say(text_input.get()),
        engine.runAndWait(),
        text_input.delete(0, 'end')
    ))


# Status Label and Log
status_label = tb.Label(frame_3, text="", font=("Segoe UI", 11))
status_label.grid(row=0, column=0, pady=10)

log = tb.ScrolledText(frame_3, height=4, width=40)
log.grid(row=1, column=0, pady=10)

# Layout adjustments: Move chart button under the input controls



def log_event(text):
    root.after(0, lambda: (log.insert(tk.END, f"{text}\n"), log.see(tk.END)))
def stop_gesture_recognition():
    global gesture_running
    gesture_running = False

def display_gesture_animation(word):
    global animation_id, frames, gif

    word = word.strip().lower()
    folder = '../animations/'
    files = [f[:-4] for f in os.listdir(folder) if f.endswith('.gif')]

    if word not in files:
        matches = difflib.get_close_matches(word, files, n=1, cutoff=0.6)
        if matches:
            word = matches[0]
        else:
            status_label.config(text="Gesture not available.")
            log_event("Gesture not available.")
            return

    filepath = os.path.join(folder, f"{word}.gif")

    try:
        gif = Image.open(filepath)
        target_size = (300, 300)
        frames = [ImageTk.PhotoImage(frame.copy().resize(target_size, Image.Resampling.LANCZOS)) for frame in ImageSequence.Iterator(gif)]

        if not frames:
            status_label.config(text="No frames in the GIF.")
            log_event("No frames in the GIF.")
            return

        if animation_id is not None:
            root.after_cancel(animation_id)

        def update(idx=0):
            global animation_id
            gif_label.config(image=frames[idx])
            animation_id = root.after(100, update, (idx + 1) % len(frames))

        update()
        status_label.config(text=f"Showing: {word}")
        log_event(f"Showing: {word}")

    except Exception as e:
        status_label.config(text=f"Error loading GIF: {e}")
        log_event(f"Error loading GIF: {e}")
async def transcribe_async(filename):
    loop = asyncio.get_event_loop()
    def blocking_transcribe():
        with whisper_lock:
            return whisper_model.transcribe(filename)
    return await loop.run_in_executor(executor, blocking_transcribe)

# Audio Recording and Processing
def record_audio():
    while running:
        audio = sd.rec(int(sample_rate * chunk_duration), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        audio_queue.put(audio.copy())
        numpy.linalg.norm(audio)
def process_audio():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while running:
        if not audio_queue.empty():
            audio = audio_queue.get()
            timestamp = int(time.time())
            filename = f"temp_audio_{timestamp}.wav"
            write(filename, sample_rate, audio)

            async def handle_transcription():
                try:
                    result = await transcribe_async(filename)
                    text = result["text"].strip().lower()
                    if text:
                        usage_stats["voice_to_gesture"] += 1
                        root.after(0, status_label.config, {"text": f"You said: {text}"})
                        root.after(0, log_event, f"You said: {text}")
                        root.after(0, display_gesture_animation, text)
                finally:
                    try:
                        os.remove(filename)
                    except:
                        pass

            loop.run_until_complete(handle_transcription())

def start_listening():
    global running
    running = True
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_audio, daemon=True).start()


def stop_listening():
    global running
    running = False

# Gesture Recognition Function
def gesture_to_text_audio():
    global gesture_running
    gesture_running = True
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    last_prediction = None
    cooldown_start = time.time()
    cooldown_period = 2  # seconds

    while gesture_running:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks)
                landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:
                    scaled = scaler.transform([landmarks])
                    prediction = model_gesture.predict(scaled)[0]

                    # Only speak if it's a new prediction or cooldown expired
                    current_time = time.time()
                    if prediction != last_prediction or (current_time - cooldown_start) > cooldown_period:
                        last_prediction = prediction
                        cooldown_start = current_time
                        usage_stats["gesture_to_text_audio"] += 1
                        root.after(0, status_label.config, {"text": f"Detected: {prediction}"})
                        root.after(0, log_event, f"Detected: {prediction}")
                        threading.Thread(target=lambda: speak_text(prediction)).start()

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# This will ensure that you correctly display the usage chart
chart_frame = tb.Frame(right_panel)
chart_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")


chart_visible = False  # Add this at the top or globally


def show_usage_chart():
    global chart_visible
    if chart_visible:
        for widget in chart_frame.winfo_children():
            widget.destroy()
        chart_visible = False
        chart_btn.config(text="📊 Show Usage Chart")
    else:
        for widget in chart_frame.winfo_children():
            widget.destroy()

        # Convert usage_stats using label_map
        custom_stats = {label_map[k]: v for k, v in usage_stats.items() if k in label_map}

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

        # Bar chart
        ax.bar(custom_stats.keys(), custom_stats.values(), color='skyblue')

        # Ensure font supports emojis
        ax.set_title("Feature Usage", fontsize=14, fontname="Arial Unicode MS")
        ax.set_xlabel("Feature", fontsize=10, fontname="Arial Unicode MS")
        ax.set_ylabel("Count", fontsize=9, fontname="Arial Unicode MS")

        # Set x-tick labels with emojis
        ax.set_xticks(range(len(custom_stats)))
        ax.set_xticklabels(custom_stats.keys(), fontsize=8, fontname="Arial Unicode MS")

        # Create the canvas for tkinter display
        chart_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        chart_visible = True
        chart_btn.config(text="❌ Hide Usage Chart")

# Buttons for UI
language_menu = tb.OptionMenu(frame_1, selected_language, *language_options, command=change_language)
language_menu.grid(row=1, column=0, padx=10, pady=5)

text_input.bind("<Return>", lambda e: (usage_stats.update({"text_to_gesture": usage_stats["text_to_gesture"] + 1}), display_gesture_animation(text_input.get().lower()), engine.say(text_input.get()), engine.runAndWait(), text_input.delete(0, 'end')))

btn1 = tb.Button(frame_1, text="\U0001F3A4 Start Voice to Gesture", command=start_listening, bootstyle=PRIMARY)
btn1.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

btn2 = tb.Button(frame_1, text="\U0001F6D1 Stop Voice Recognition", command=stop_listening, bootstyle=DANGER)
btn2.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

btn3 = tb.Button(frame_1, text="\u2328\ufe0f Text to Gesture", command=lambda: (usage_stats.update({"text_to_gesture": usage_stats["text_to_gesture"] + 1}), display_gesture_animation(text_input.get().lower())), bootstyle=INFO)
btn3.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

btn4 = tb.Button(frame_1, text="\U0001F590 Gesture to Text/Audio", command=lambda: threading.Thread(target=gesture_to_text_audio).start(), bootstyle=SUCCESS)
btn4.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
chart_btn = tb.Button(right_panel, text="\U0001F4CA Show Usage Chart", command=show_usage_chart, bootstyle=INFO)
chart_btn.grid(row=1, column=1, sticky="ne", padx=10, pady=5)
# Theme toggle

btn_stop_gesture = tb.Button(frame_1, text="🛑 Stop Gesture Recognition", command=stop_gesture_recognition, bootstyle=DANGER)
btn_stop_gesture.grid(row=4, column=1, padx=10, pady=5, sticky="ew")
tb.Label(right_panel, text="Text Input", font=("Segoe UI", 8, "bold")).grid(row=0, column=0, sticky="n", pady=(5, 0))

def toggle_theme():
    current_theme = root.style.theme.name
    new_theme = "darkly" if current_theme == "flatly" else "flatly"
    root.style.theme_use(new_theme)
    theme_btn = tb.Button(frame_1, text="🌓 Toggle Theme", command=toggle_theme, bootstyle=SECONDARY)
    theme_btn.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

def on_closing():
    global running, gesture_running
    running = False
    gesture_running = False
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)





root.mainloop()