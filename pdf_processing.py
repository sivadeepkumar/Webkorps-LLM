from PyPDF2 import PdfReader

pdf_filenames = ["Groups_and_Records.pdf"] 

raw_text = ""

# Loop through PDFs and extract text
for pdf_filename in pdf_filenames:
    pdfreader = PdfReader(pdf_filename)
    for page in pdfreader.pages:
        content = page.extract_text() 
        if content:
            raw_text += content

# Do something with extracted text like save it to a file
with open('./sample_data/text_samples/extracted_text.txt', 'w') as f:
    f.write(raw_text)





pdf_filenames = ["Leave_and_attendance_policy_2024.pdf"] 

raw_text = ""

# Loop through PDFs and extract text
for pdf_filename in pdf_filenames:
    pdfreader = PdfReader(pdf_filename)
    for page in pdfreader.pages:
        content = page.extract_text() 
        if content:
            raw_text += content

# Do something with extracted text like save it to a file
with open('./sample_data/text_samples/webkorps_data.txt', 'w') as f:
    f.write(raw_text)

