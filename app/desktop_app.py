

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import requests

API_URL = "http://127.0.0.1:5000/predict"
EXPLAIN_URL = "http://127.0.0.1:5000/explain"
HEALTH_URL = "http://127.0.0.1:5000/health"


class PhishingDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Email Detector")
        self.root.geometry("640x700")
        self.root.resizable(False, False)

        # Title 
        title = tk.Label(
            root, text="Phishing Email Detector",
            font=("Helvetica", 18, "bold")
        )
        title.pack(pady=(15, 0))

        subtitle = tk.Label(
            root, text="Paste an email's subject + body below and click Check.",
            font=("Helvetica", 10), fg="gray30"
        )
        subtitle.pack(pady=(0, 10))

        # Input box 
        self.text_input = scrolledtext.ScrolledText(
            root, width=70, height=15, font=("Helvetica", 11), wrap=tk.WORD
        )
        self.text_input.pack(padx=15, pady=5)

        # Button
        self.check_button = tk.Button(
            root, text="Check Email", font=("Helvetica", 12, "bold"),
            bg="#2E5090", fg="white", activebackground="#1E3A66",
            command=self.on_check_clicked, padx=20, pady=8
        )
        self.check_button.pack(pady=12)

        # Result display 
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=5)

        self.result_label = tk.Label(
            self.result_frame, text="", font=("Helvetica", 16, "bold")
        )
        self.result_label.pack()

        self.prob_label = tk.Label(
            self.result_frame, text="", font=("Helvetica", 11), fg="gray30"
        )
        self.prob_label.pack()

        # XAI explanation panel
        explain_title = tk.Label(
            root, text="Why? (word-level contributions)",
            font=("Helvetica", 10, "bold"), fg="gray30"
        )
        explain_title.pack(pady=(8, 2))

        self.explanation_box = tk.Text(
            root, width=70, height=6, font=("Helvetica", 10),
            state=tk.DISABLED, bg="#F7F7F7", wrap=tk.WORD
        )
        self.explanation_box.pack(padx=15, pady=(0, 5))
        self.explanation_box.tag_configure("phishing_word", foreground="#C44E52", font=("Helvetica", 10, "bold"))
        self.explanation_box.tag_configure("legit_word", foreground="#2E5090", font=("Helvetica", 10, "bold"))
        self.explanation_box.tag_configure("plain", foreground="gray30")

        # Status bar
        self.status_var = tk.StringVar(value="Checking connection to API...")
        status_bar = tk.Label(
            root, textvariable=self.status_var, font=("Helvetica", 9),
            fg="gray50", anchor="w"
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Check API availability at startup (non-blocking)
        threading.Thread(target=self.check_api_health, daemon=True).start()

    def check_api_health(self):
        try:
            resp = requests.get(HEALTH_URL, timeout=3)
            if resp.status_code == 200:
                self.status_var.set("Connected to model API.")
            else:
                self.status_var.set("API reachable but returned an error.")
        except requests.exceptions.RequestException:
            self.status_var.set(
                "Could not reach the API. Make sure api.py is running on http://127.0.0.1:5000"
            )

    def on_check_clicked(self):
        email_text = self.text_input.get("1.0", tk.END).strip()
        if not email_text:
            messagebox.showwarning("Empty Input", "Please paste some email text first.")
            return

        self.check_button.config(state=tk.DISABLED, text="Checking...")
        self.result_label.config(text="")
        self.prob_label.config(text="")
        self._set_explanation_text("Analysing...", [])

        # Running API call in a background thread so UI doesn't freeze
        threading.Thread(target=self.call_api, args=(email_text,), daemon=True).start()

    def call_api(self, email_text):
        try:
            # /explain returns label + probability + LIME word-level contributions
            response = requests.post(EXPLAIN_URL, json={"text": email_text, "num_features": 10}, timeout=20)
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self.display_result, result)
            else:
                error_msg = response.json().get("error", "Unknown error from API.")
                self.root.after(0, self.display_error, error_msg)
        except requests.exceptions.RequestException as e:
            self.root.after(0, self.display_error, f"Could not connect to API: {e}")
        finally:
            self.root.after(0, lambda: self.check_button.config(state=tk.NORMAL, text="Check Email"))

    def display_result(self, result):
        label = result["label"]
        probability = result["probability"]
        explanation = result.get("explanation", [])

        if label == "Phishing":
            self.result_label.config(text="⚠ PHISHING", fg="#C44E52")
        else:
            self.result_label.config(text="✓ LEGITIMATE", fg="#2E8B57")

        self.prob_label.config(text=f"Phishing probability: {probability:.2%}")
        self._set_explanation_text(None, explanation)

    def _set_explanation_text(self, placeholder, explanation):
        """Render LIME word contributions, colour-coded by direction."""
        self.explanation_box.config(state=tk.NORMAL)
        self.explanation_box.delete("1.0", tk.END)

        if placeholder:
            self.explanation_box.insert(tk.END, placeholder, "plain")
        elif not explanation:
            self.explanation_box.insert(tk.END, "No explanation available.", "plain")
        else:
            # Sort by absolute contribution, strongest first
            explanation_sorted = sorted(explanation, key=lambda wv: abs(wv[1]), reverse=True)
            self.explanation_box.insert(
                tk.END,
                "Red = pushes toward Phishing   |   Blue = pushes toward Legitimate\n\n",
                "plain"
            )
            for word, weight in explanation_sorted:
                tag = "phishing_word" if weight > 0 else "legit_word"
                self.explanation_box.insert(tk.END, f"{word}  ({weight:+.3f})    ", tag)

        self.explanation_box.config(state=tk.DISABLED)

    def display_error(self, message):
        messagebox.showerror("Error", message)
        self.status_var.set(message)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingDetectorApp(root)
    root.mainloop()
