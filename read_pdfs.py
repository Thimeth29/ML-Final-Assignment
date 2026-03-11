import PyPDF2
import os

pdf_dir = "ASG"
files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
with open('pdf_contents.txt', 'w', encoding='utf-8') as out:
    for f in files:
        out.write(f"--- {f} ---\n")
        try:
            reader = PyPDF2.PdfReader(os.path.join(pdf_dir, f))
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    out.write(f"Page {i+1}:\n{text}\n")
        except Exception as e:
            out.write(f"Error reading {f}: {e}\n")
