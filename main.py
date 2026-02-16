# Copyright 2026 The OpenSLM Project
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import sentencepiece as spm

class MuseSEO:
    def __init__(self, tokenizer_path="models/ecommerce_tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    def generate_product_description(self, product_name, attributes, sources):
        print(f"MUSE: Génération de contenu SEO pour {product_name}")
        
        description = f"Découvrez notre {product_name}. "
        description += f"Conçu en {attributes.get('material', 'matériaux premium')}, "
        description += f"ce modèle pèse environ {attributes.get('weight', 'N/A')}."
        
        # Ajout des citations style Perplexity
        if sources:
            description += "

Sources : " + ", ".join(sources)
            
        return description

if __name__ == "__main__":
    muse = MuseSEO()
    print(muse.generate_product_description(
        "Nike Air Max", 
        {"material": "Cuir", "weight": "850g"}, 
        ["https://nike.com", "ScyllaDB Local"]
    ))