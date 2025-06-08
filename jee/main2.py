import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import streamlit as st
import io
import PyPDF2

# Set page config
st.set_page_config(page_title="Edumorph", page_icon="📚", layout="wide")

# API key
API_KEY = "Your_gemini_api_key"

# Initialize session state variables if they don't exist
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'quiz_result' not in st.session_state:
    st.session_state.quiz_result = ""

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('resources.csv')
    jee_resources = data['Subject'] + ' ' + data['Topic'] + ' ' + data['Resource Name'] + ' ' + data['Resource Link']
    vectorizer = TfidfVectorizer()
    jee_data = vectorizer.fit_transform(jee_resources)
    return data, jee_data, vectorizer

# Recommendation function
def recommend(input_topic, data, jee_data):
    list_of_topics = data['Topic'].tolist()
    close_matches = difflib.get_close_matches(input_topic, list_of_topics)
    
    if not close_matches:
        return "No matching topics found. Please try a different topic."
    
    index_of_topic = data[data.Topic == close_matches[0]].index.values[0]
    similarity_matrix = cosine_similarity(jee_data)
    
    my_similarity = list(enumerate(similarity_matrix[index_of_topic]))
    sorted_similar_topic = sorted(my_similarity, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, topic in enumerate(sorted_similar_topic[:5], 1):
        index = topic[0]
        topic_from_index = data[data.index == index]['Topic'].values[0]
        subject_from_index = data[data.index == index]['Subject'].values[0]
        resource_name_from_index = data[data.index == index]['Resource Name'].values[0]
        resource_link_from_index = data[data.index == index]['Resource Link'].values[0]
        
        recommendations.append({
            'rank': i,
            'topic': topic_from_index,
            'subject': subject_from_index,
            'resource_name': resource_name_from_index,
            'resource_link': resource_link_from_index
        })
    
    return recommendations

# Summarization function
@st.cache_resource
def load_summarization_model():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def summarize(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Question answering function
@st.cache_resource
def load_qa_model():
    model = AutoModelForQuestionAnswering.from_pretrained('deepset/roberta-base-squad2')
    tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')
    return model, tokenizer

def question_answer(question, context, model, tokenizer):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    output = model(**inputs)
    
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    
    # Check if the answer makes sense (start <= end)
    if answer_start <= answer_end:
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end+1])
        )
    else:
        answer = "Unable to find an answer in the provided context."
    
    return answer

# Quiz function
def generate_quiz(topic):
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.8, google_api_key=API_KEY)
    prompt = PromptTemplate(
        template='Generate a 5 question quiz on {topic} with multiple choice options (A, B, C, D) and provide the correct answers at the end.',
        input_variables=['topic']
    )
    parser = StrOutputParser()
    chain = prompt | model | parser
    result = chain.invoke({"topic": topic})
    return result

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Main app
def main():
    st.title("📚 Edumorph")
    st.subheader("Your personal learning assistant")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio(
        "Choose a feature:",
        ["Resource Recommender", "PDF Summarizer & Q&A", "Quiz Generator"]
    )
    
    # Load data
    data, jee_data, vectorizer = load_data()
    
    # Resource Recommender
    if option == "Resource Recommender":
        st.header("📖 Resource Recommender")
        st.markdown("Find the best study resources for your topics")
        
        # Get unique subjects for dropdown
        subjects = sorted(data['Subject'].unique())
        selected_subject = st.selectbox("Select a subject", subjects)
        
        # Filter topics by selected subject
        filtered_topics = sorted(data[data['Subject'] == selected_subject]['Topic'].unique())
        
        # Let user select from filtered topics or enter custom topic
        topic_selection_method = st.radio("Topic selection method:", ["Select from list", "Enter custom topic"])
        
        if topic_selection_method == "Select from list":
            selected_topic = st.selectbox("Select a topic", filtered_topics)
        else:
            selected_topic = st.text_input("Enter a topic")
        
        if st.button("Get Recommendations"):
            if selected_topic:
                with st.spinner("Finding resources..."):
                    recommendations = recommend(selected_topic, data, jee_data)
                
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.success(f"Found {len(recommendations)} recommendations for {selected_topic}")
                    
                    for rec in recommendations:
                        with st.expander(f"{rec['rank']}. {rec['resource_name']} ({rec['subject']} - {rec['topic']})"):
                            st.markdown(f"**Topic:** {rec['topic']}")
                            st.markdown(f"**Subject:** {rec['subject']}")
                            st.markdown(f"**Resource:** {rec['resource_name']}")
                            st.markdown(f"**Link:** [{rec['resource_link']}]({rec['resource_link']})")
            else:
                st.warning("Please enter or select a topic")
    
    # PDF Summarizer & Q&A
    elif option == "PDF Summarizer & Q&A":
        st.header("📝 PDF Summarizer & Q&A")
        st.markdown("Upload a PDF to summarize and ask questions about it")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            # Save the uploaded file
            pdf_bytes = io.BytesIO(uploaded_file.read())
            
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(pdf_bytes)
                text_preview = text[:500] + "..." if len(text) > 500 else text
                
                with st.expander("Preview extracted text"):
                    st.write(text_preview)
            
            # Load models
            summarization_model, summarization_tokenizer = load_summarization_model()
            
            # Summarize button
            if st.button("Summarize PDF"):
                with st.spinner("Summarizing content..."):
                    summary = summarize(text, summarization_model, summarization_tokenizer)
                    st.session_state.summary = summary
                
                st.subheader("Summary")
                st.write(st.session_state.summary)
            
            # Display previously generated summary if available
            if st.session_state.summary:
                st.subheader("Summary")
                st.write(st.session_state.summary)
                
                # Load QA model
                qa_model, qa_tokenizer = load_qa_model()
                
                # Question answering section
                st.subheader("Ask a question about the document")
                question = st.text_input("Your question")
                
                if st.button("Get Answer") and question:
                    with st.spinner("Finding answer..."):
                        answer = question_answer(question, st.session_state.summary, qa_model, qa_tokenizer)
                    
                    st.success("Answer found")
                    st.markdown(f"**Q: {question}**")
                    st.markdown(f"**A: {answer}**")
    
    # Quiz Generator
    elif option == "Quiz Generator":
        st.header("🧠 Quiz Generator")
        st.markdown("Generate a quiz on any topic to test your knowledge")
        
        quiz_topic = st.text_input("Enter the topic for your quiz")
        
        if st.button("Generate Quiz") and quiz_topic:
            with st.spinner("Generating quiz..."):
                quiz_result = generate_quiz(quiz_topic)
                st.session_state.quiz_result = quiz_result
            
            st.success(f"Quiz on {quiz_topic} generated!")
            st.markdown(st.session_state.quiz_result)
        
        # Display previously generated quiz if available
        elif st.session_state.quiz_result:
            st.markdown(st.session_state.quiz_result)

if __name__ == "__main__":
    main()
