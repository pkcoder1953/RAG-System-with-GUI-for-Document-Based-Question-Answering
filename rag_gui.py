from typing import List, Dict
import os
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import hashlib
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import io

class RAGSystem:
    def __init__(self, cache_file="query_cache.json"):
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        self.document_chunks = []
        self.document_embeddings = None
        self.image_store = {}
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, default=convert_to_serializable)

    def _generate_cache_key(self, query: str, context: str) -> str:
        content = f"{query}|{context}"
        return hashlib.md5(content.encode()).hexdigest()

    def process_pdf(self, pdf_path: str, chunk_size: int = 1000):
        print(f"Processing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        current_chunk = ""
        current_page = 0
        for page_num, page in enumerate(doc):
            text = page.get_text()
            current_chunk += text
            if len(current_chunk) >= chunk_size:
                self.document_chunks.append({
                    'text': current_chunk,
                    'page': current_page,
                    'end_page': page_num,
                    'source': pdf_path
                })
                current_chunk = ""
                current_page = page_num + 1
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                surrounding_text = page.get_text()
                image_id = f"page_{page_num}_img_{img_index}"
                self.image_store[image_id] = {
                    'image_data': image_data,
                    'context': surrounding_text,
                    'page': page_num,
                    'source': pdf_path
                }
        if current_chunk:
            self.document_chunks.append({
                'text': current_chunk,
                'page': current_page,
                'end_page': len(doc) - 1,
                'source': pdf_path
            })
        texts = [chunk['text'] for chunk in self.document_chunks]
        self.document_embeddings = self.embeddings_model.encode(texts)
        doc.close()
        print(f"Finished processing PDF. Extracted {len(self.document_chunks)} chunks and {len(self.image_store)} images.")

    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embeddings_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.document_chunks[i] for i in top_indices]

    def find_relevant_images(self, answer_text: str, similarity_threshold: float = 0.5) -> List[Dict]:
        if not self.image_store:
            return []
        answer_embedding = self.embeddings_model.encode([answer_text])[0]
        relevant_images = []
        for image_id, image_info in self.image_store.items():
            context_embedding = self.embeddings_model.encode([image_info['context']])[0]
            similarity = cosine_similarity([answer_embedding], [context_embedding])[0][0]
            if similarity >= similarity_threshold:
                relevant_images.append({
                    'image_id': image_id,
                    'page': image_info['page'],
                    'source': image_info['source'],
                    'similarity': similarity
                })
        return sorted(relevant_images, key=lambda x: x['similarity'], reverse=True)

    def generate_answer(self, query: str) -> Dict:
        relevant_chunks = self.find_relevant_chunks(query)
        context = "\n".join([chunk['text'] for chunk in relevant_chunks])
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < 86400:
                print("(Retrieved from cache)")
                return cached_result['result']
        prompt = f"Answer the question: {query} based on the following context:\n\n{context[:1000]}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(inputs.input_ids, max_length=512, num_beams=4, temperature=0.7, top_p=0.9, do_sample=True)
        answer_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        relevant_images = self.find_relevant_images(answer_text)
        result = {
            'answer': answer_text,
            'images': relevant_images,
            'source_pages': [(chunk['page'], chunk['source']) for chunk in relevant_chunks]
        }
        self.cache[cache_key] = {'result': result, 'timestamp': time.time()}
        self._save_cache()
        return result

def display_result_in_gui(answer, source_pages, images):
    root = tk.Tk()
    root.title("Query Result")
    answer_frame = ttk.Frame(root, padding="10")
    answer_frame.grid(row=0, column=0, sticky="nsew")
    ttk.Label(answer_frame, text="Answer:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
    answer_text = scrolledtext.ScrolledText(answer_frame, wrap=tk.WORD, width=60, height=10, font=("Arial", 12))
    answer_text.insert(tk.END, answer)
    answer_text.configure(state="disabled")
    answer_text.grid(row=1, column=0, sticky="nsew", pady=5)
    source_frame = ttk.Frame(root, padding="10")
    source_frame.grid(row=1, column=0, sticky="nsew")
    ttk.Label(source_frame, text="Source Pages:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
    source_text = scrolledtext.ScrolledText(source_frame, wrap=tk.WORD, width=60, height=5, font=("Arial", 12))
    for page, source in source_pages:
        source_text.insert(tk.END, f"- Page {page} in {source}\n")
    source_text.configure(state="disabled")
    source_text.grid(row=1, column=0, sticky="nsew", pady=5)
    image_frame = ttk.Frame(root, padding="10")
    image_frame.grid(row=2, column=0, sticky="nsew")
    ttk.Label(image_frame, text="Relevant Images:", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
    if images:
        for idx, image_info in enumerate(images):
            image_id = image_info['image_id']
            image_data = rag.image_store[image_id]['image_data']
            pil_image = Image.open(io.BytesIO(image_data))
            pil_image.thumbnail((200, 200))
            tk_image = ImageTk.PhotoImage(pil_image)
            img_label = ttk.Label(image_frame, image=tk_image)
            img_label.image = tk_image
            img_label.grid(row=(idx // 3) + 1, column=idx % 3, padx=5, pady=5)
    else:
        ttk.Label(image_frame, text="No relevant images found.", font=("Arial", 12)).grid(row=1, column=0, sticky="w")
    root.mainloop()

rag = RAGSystem()
pdf_directory = "./datapdf"
if not os.path.exists(pdf_directory):
    print(f"Error: Directory '{pdf_directory}' not found!")
else:
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    for pdf_file in pdf_files:
        rag.process_pdf(str(pdf_file))

while True:
    query = input("\nEnter your question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    result = rag.generate_answer(query)
    display_result_in_gui(result['answer'], result['source_pages'], result['images'])
