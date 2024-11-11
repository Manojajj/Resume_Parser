import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PyPDF2 import PdfReader

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # Clean up the extracted text
        text = " ".join(text.split())  # Replace multiple spaces and newlines with a single space
        text = text.replace('\n', ' ')  # Replace newline characters with a space
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
    return text


def parse_resume(pdf_file):
    resume_text = extract_text_from_pdf(pdf_file)
    if not resume_text.strip():
        return {"error": "No text extracted from the resume."}

    # Define a more detailed prompt
    prompt = (
        "You are a smart resume parser. Your task is to extract key information from the provided resume text and present it in a structured JSON format. "
        "Please carefully identify and extract the following details:\n"
        "- Phone Number (e.g., 1234567890 or formatted with spaces)\n"
        "- Email Address (e.g., example@example.com)\n"
        "- LinkedIn Profile URL (e.g., linkedin.com/in/username)\n"
        "- GitHub Profile URL (e.g., github.com/username)\n"
        "- Certifications (list all certifications mentioned)\n"
        "- Projects (provide details of projects worked on)\n"
        "- Technical Skills (list all relevant skills)\n"
        "- Education (list educational qualifications)\n\n"
        "Here is the resume text for extraction:\n{resume_text}\n\n"
        "Please provide the output as a JSON object."
    ).format(resume_text=resume_text)

    # Tokenize and truncate the input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    if input_ids.size(1) > 1024:
        input_ids = input_ids[:, :1024]

    # Generate output from the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=500, num_return_sequences=1)

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("Model Output:", output_text)  # Debugging statement

    # Initialize the structured output
    structured_output = {
        "Certifications": [],
        "Education": [],
        "Email": None,
        "GitHub": None,
        "LinkedIn": None,
        "Phone": None,
        "Projects": [],
        "Technical Skills": []
    }

    # Process output text to make it structured
    lines = output_text.splitlines()
    for line in lines:
        line = line.strip()
        if "Phone" in line:
            structured_output["Phone"] = line.split("Phone")[-1].strip()
        elif "Email" in line:
            structured_output["Email"] = line.split("Email")[-1].strip()
        elif "LinkedIn" in line:
            structured_output["LinkedIn"] = line.split("LinkedIn")[-1].strip()
        elif "GitHub" in line:
            structured_output["GitHub"] = line.split("GitHub")[-1].strip()
        elif "Certifications" in line:
            structured_output["Certifications"] = line.split("Certifications")[-1].strip().split(',')
        elif "Projects" in line:
            structured_output["Projects"] = line.split("Projects")[-1].strip().split(',')
        elif "Technical Skills" in line:
            structured_output["Technical Skills"] = line.split("Technical Skills")[-1].strip().split(',')
        elif "Education" in line:
            structured_output["Education"] = line.split("Education")[-1].strip().split(',')

    # Return the structured output
    return {"parsed_data": structured_output}
