# 🧠 JEE Adaptive Learning Platform

An AI-powered adaptive learning platform tailored for JEE aspirants. The system dynamically adjusts the learning path based on students’ real-time performance, delivering the right content at the right time to maximize retention and understanding.

---

## 🚀 Features

- 🔐 **User Registration & Profile Management**  
  Personalized dashboard for each student with performance tracking.

- 📊 **Adaptive Quiz Generator**  
  Quizzes generated based on user proficiency using GPT-based models.

- 📈 **Real-Time Proficiency Analysis**  
  Tracks student performance topic-wise and adjusts content difficulty.

- 🎯 **Personalized Learning Paths**  
  Recommends topics and learning resources (videos + articles) based on student weaknesses and progress.

- 📚 **JEE Syllabus-Centric Content**  
  Covers Math, Physics, and Chemistry topics aligned with the JEE Main/Advanced syllabus.

- 💡 **Smart Resource Recommendation Engine**  
  Recommends curated YouTube videos, notes, and reading materials using NLP and cosine similarity-based matching.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Database**: SQL
- **AI/ML**: 
  -  Gemini API
  - Scikit-learn (for profiling & recommendations)
  - Transformers (for NER and summarization)
- **Authentication**: Firebase Auth / JWT
- **Deployment**: Vercel / Render / Railway

---

## 🧩 Architecture Overview

1. **User logs in** and starts a diagnostic test.
2. The system evaluates responses and **profiles performance**.
3. Based on proficiency, a **personalized quiz** is generated using GPT or rules-based logic.
4. Scores are stored and analyzed.
5. Using vector similarity and metadata, **resources are recommended** for improvement.
6. User receives a **personalized dashboard** and updated learning path.

---

## 📦 Installation & Setup

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/jee-adaptive-learning.git
cd jee-adaptive-learning

# 2. Install dependencies
npm install

# 3. Set up environment variables
# Create a `.env` file and add:
# OPENAI_API_KEY=your_openai_key
# MONGODB_URI=your_mongodb_connection_string
# JWT_SECRET=your_jwt_secret

# 4. Run the app
npm run dev
