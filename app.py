from flask import Flask, request, jsonify, render_template
import base64
import io
from generator import NakedMaleGenerator

app = Flask(__name__)
gen = NakedMaleGenerator()

BODY_PRESETS = {
    "lean": "slim lean male body, toned, minimal body fat, defined",
    "athletic": "athletic male build, visible muscle definition, fit",
    "muscular": "highly muscular male, bodybuilder physique, large muscles, veiny",
    "stocky": "stocky male build, broad shoulders, thick frame",
    "heavyset": "heavyset male, larger body, soft midsection, big frame",
    "twink": "young slim male, smooth skin, petite frame, androgynous",
    "bear": "hairy muscular male, thick chest hair, beard, stocky",
    "dadbody": "dad body male, average build, slight belly, natural"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        body_type = data.get("body_type", "athletic")
        body_desc = BODY_PRESETS.get(body_type, BODY_PRESETS["athletic"])
        custom_prompt = data.get("custom_prompt", "")
        prompt = f"full body nude male, {body_desc}, {custom_prompt}, realistic photography, studio lighting, detailed skin texture, anatomically correct"
        negative_prompt = data.get("negative_prompt", "clothed, underwear, deformed, bad anatomy, blurry, low quality")
        width = int(data.get("width", 1024))
        height = int(data.get("height", 1024))
        steps = int(data.get("steps", 50))
        guidance_scale = float(data.get("guidance_scale", 7.5))
        seed = data.get("seed")
        seed = int(seed) if seed else None
        batch_count = int(data.get("batch_count", 1))

        images, seeds = gen.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_count=batch_count
        )

        results = []
        for img, s in zip(images, seeds):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            results.append({"image": img_b64, "seed": s, "prompt": prompt})

        return jsonify({"success": True, "results": results, "count": len(results)})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
