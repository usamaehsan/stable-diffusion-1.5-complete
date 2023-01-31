import banana_dev as banana

model_inputs = {
    "prompt": "Super Bnana, savior of the world",
    "negative_prompt": "boring superman",
    "height": 512,
    "width": 512,
    "steps": 50,
    "guidance_scale": 9,
    "seed": None,
    "scheduler": "K_EULER_ANCESTRAL"
}

api_key = "YOUR_API_KEY"
model_key = "YOUR_MODEL_KEY"

# Run the model
out = banana.run(api_key, model_key, model_inputs)
