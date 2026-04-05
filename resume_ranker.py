import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
import spacy
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")


@lru_cache(maxsize=1)
def get_nlp_model():
    # Lazy-load to reduce app startup latency.
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def get_bert_model():
    # Lazy-load to avoid heavy model initialization at import time.
    return SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()

# Function to extract text from image (OCR)
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

# Function to get BERT embeddings
def get_embeddings(text):
    return get_bert_model().encode(text, convert_to_tensor=True)

# Function to calculate similarity score
def calculate_similarity(resume_embedding, job_embedding):
    return util.pytorch_cos_sim(resume_embedding, job_embedding).item()



# Function to extract key details from resumes using Named Entity Recognition (NER)
def extract_resume_details(text):
    doc = get_nlp_model()(text)
    details = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Skills": [],
        "Experience": None,
        "Education": None
    }
    
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not details["Name"]:
            details["Name"] = ent.text
        elif ent.label_ in ["ORG", "WORK_OF_ART"]:
            details["Education"] = ent.text
        elif ent.label_ == "DATE":
            details["Experience"] = ent.text

    email_match = EMAIL_REGEX.search(text)
    if email_match:
        details["Email"] = email_match.group(0)

    phone_match = PHONE_REGEX.search(text)
    if phone_match:
        details["Phone"] = phone_match.group(0)
            
    return details



# Function to rank resumes based on job description
def rank_resumes(resumes, job_description):
    job_embedding = get_embeddings(job_description)
    ranked_resumes = []

    for resume in resumes:
        filename = resume.name.lower()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(resume)
        elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
            text = extract_text_from_image(resume)
        else:
            st.warning(f"Skipping unsupported file type: {resume.name}")
            continue

        if not text.strip():
            st.warning(f"No readable text found in: {resume.name}")
            continue

        resume_embedding = get_embeddings(text)
        score = calculate_similarity(resume_embedding, job_embedding)
        details = extract_resume_details(text)
        ranked_resumes.append((details, score))
    
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
    return ranked_resumes



# Function to check diversity and bias in ranking
def check_diversity(ranked_resumes):
    if not ranked_resumes:
        return "No resumes were ranked, so diversity could not be assessed."
    return "Bias check is currently a placeholder. Add a measurable fairness metric before using this in production."



st.title("📄 AI-Powered Resume Screening & Ranking System")

# Upload resumes
uploaded_resumes = st.file_uploader("Upload Resumes (PDF or Image)", accept_multiple_files=True)

# Upload job description
job_description = st.text_area("Enter Job Description (or Upload as File)")

# Button to start ranking
if st.button("Analyze & Rank Resumes"):
    if uploaded_resumes and job_description:
        with st.spinner("Processing resumes..."):
            ranked_resumes = rank_resumes(uploaded_resumes, job_description)
            bias_message = check_diversity(ranked_resumes)

            # Display results
            st.subheader("📊 Ranked Resumes")
            if not ranked_resumes:
                st.info("No valid resumes could be processed. Please upload supported files with readable text.")
            for i, (details, score) in enumerate(ranked_resumes):
                st.write(f"**Rank {i+1}: {details.get('Name', 'Unknown')}**")
                st.write(f"🔹 **Score:** {round(score * 100, 2)}% match")
                st.write(f"📧 **Email:** {details.get('Email', 'N/A')}")
                st.write(f"📞 **Phone:** {details.get('Phone', 'N/A')}")
                st.write(f"🎓 **Education:** {details.get('Education', 'N/A')}")
                st.write(f"💼 **Experience:** {details.get('Experience', 'N/A')}")
                st.write("—" * 30)
            
            # Show bias check
            st.subheader("⚖️ Diversity & Bias Check")
            st.write(bias_message)
    else:
        st.error("Please upload resumes and enter a job description!")

# Display footer
st.markdown("---")
st.write("Built with ❤️ by [Rushikesh Bobade]")
st.write("[GitHub](https://github.com/rushikesh369/AI-resume-screening-system.git) | [LinkedIn](https://www.linkedin.com/in/rushikesh-bobade-96a69429b/)")
