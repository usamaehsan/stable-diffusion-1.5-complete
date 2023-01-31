import banana_dev as banana

model_inputs = {
    "prompt": "Super Bnana, savior of the world",
    "negative_prompt": "boring superman",
    "height": 512,
    "width": 512,
    "steps": 20,
    "guidance_scale": 9,
    "seed": None,
    "scheduler": "K_EULER_ANCESTRAL"
}

api_key = "4de3a913-c09e-4d1f-8e0e-b5c28dfa48f1"
model_key = "4afe856d-c072-4979-9bcd-f688a7d99257"

# Run the model
out = banana.run(api_key, model_key, model_inputs)
