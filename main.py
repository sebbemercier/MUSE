from fastapi import FastAPI
import httpx
import torch
from transformers import pipeline

app = FastAPI(title="MUSE | La Plume SEO")

ATLAS_URL = "http://localhost:8001"

print("Chargement du SLM MUSE (TinyLlama)...")
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

@app.get("/generate-copy/{product_id}")
async def generate_copy(product_id: int):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{ATLAS_URL}/product/{product_id}")
        product = response.json()

    if "error" in product: return product

    prompt = f"<|system|>\nTu es MUSE, une rédactrice SEO de talent. Écris une méta-description accrocheuse de moins de 160 caractères pour ce produit.</s>\n<|user|>\nProduit: {product['name']}\nDescription: {product['description']}</s>\n<|assistant|>\n"
    
    outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    copy = outputs[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
    
    return {
        "product": product["name"],
        "seo_copy": copy
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
