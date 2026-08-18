from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import uuid


app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMAGE_SIZE = (150,150)  


with open("brain.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("brain.h5")

loaded_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


class_names = ['No-tumor','Pituitary','Meningioma','Glioma']


users = {}


@app.route("/", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users:
            return "User already exists. Please sign in."

        users[username] = password
        return redirect(url_for("signin"))

    return render_template("signup.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if users.get(username) == password:
            session["username"] = username
            return redirect(url_for("home"))

        return "Invalid credentials"

    return render_template("signin.html")


@app.route("/home")
def home():
    if "username" not in session:
        return redirect(url_for("signin"))
    return render_template("home.html")


@app.route("/detection", methods=["GET", "POST"])
def detection():
    if "username" not in session:
        return redirect(url_for("signin"))

    predicted_class = None
    suggestion = None
    image_path = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if image_file and image_file.filename:
            filename = f"{uuid.uuid4().hex}_{image_file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_file.save(image_path)

            
            test_image = image.load_img(image_path, target_size=IMAGE_SIZE)
            test_image_array = image.img_to_array(test_image) / 255.0
            test_image_array = np.expand_dims(test_image_array, axis=0)

            predictions = loaded_model.predict(test_image_array)
            pred_label = np.argmax(predictions, axis=1)[0]
            predicted_class = class_names[pred_label]

            suggestions = {
                "No-tumor": (
                    "✅ No brain tumor detected.\n"
                    "• MRI scan appears normal\n"
                    "• Continue routine health checkups\n"
                    "• Consult a neurologist if symptoms persist"
                ),

                "Pituitary": (
                    "🟠 Pituitary tumor detected.\n"
                    "• Hormonal evaluation recommended\n"
                    "• Consult an endocrinologist and neurosurgeon\n"
                    "• Treatment may include medication or surgery"
                ),

                "Meningioma": (
                    "🟡 Meningioma detected.\n"
                    "• Usually slow-growing and often benign\n"
                    "• Regular MRI monitoring advised\n"
                    "• Surgery or radiation may be required if symptomatic"
                ),

                "Glioma": (
                    "🚨 Glioma detected.\n"
                    "• Can be aggressive and fast-growing\n"
                    "• Immediate consultation with neuro-oncologist required\n"
                    "• Treatment may include surgery, chemotherapy, and radiotherapy"
                )
            }

            suggestion = suggestions.get(
                predicted_class,
                "⚠️ Unable to determine result. Please consult a doctor."
            )

  
    return render_template(
        "detection.html",
        predicted_class=predicted_class,
        suggestion=suggestion,
        image_path=image_path
    )


@app.route("/model_info")
def model_info():
    if "username" not in session:
        return redirect(url_for("signin"))
    return render_template("model_info.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("signin"))



if __name__ == "__main__":
    app.run()
