import re
from collections import Counter

import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.write("Match your resume with job descriptions instantly!")

JOB_TITLE_SUGGESTIONS = [
    "Python Developer",
    "Software Developer",
    "Software Engineer",
    "Software Architect",
    "Software Tester",
    "Software Project Manager",
    "Software Quality Analyst",
    "Software Support Specialist",
    "Software Release Manager",
    "Software Automation Engineer",
    "Software Technical Lead",
    "Data Analyst",
    "Data Scientist",
    "Data Engineer",
    "Machine Learning Engineer",
    "Business Analyst",
    "Web Developer",
    "Full Stack Developer",
    "Backend Developer",
    "Frontend Developer",
    "DevOps Engineer",
    "System Administrator",
    "Database Administrator",
    "UI/UX Designer",
    "Product Manager",
    "QA Engineer",
]

JOB_DESCRIPTION_TEMPLATES = {
    "Python Developer": (
        "We are seeking highly motivated and talented Python Developers to join our dynamic team. "
        "Our team consists of professionals from diverse technical backgrounds and areas of subject expertise, "
        "responsible for aggregating, transforming, analyzing, and visualizing large volumes of web data to help our customers "
        "make smarter, data-driven decisions. We are looking for individuals with creativity, strong technical skills, keen attention "
        "to detail, and a passion for leveraging data to drive business growth. As a Python Developer, you will be involved in "
        "developing robust applications using Python frameworks such as Django, Flask, and FastAPI. You will work on data "
        "processing pipelines, implement machine learning models, and ensure the scalability and performance of our systems. "
        "Key responsibilities include writing clean, efficient code, collaborating with cross-functional teams, conducting code "
        "reviews, and participating in agile development processes. You will have the opportunity to work with cutting-edge "
        "technologies like cloud computing (AWS, GCP), containerization (Docker, Kubernetes), and big data tools (Hadoop, Spark). "
        "We value problem-solving abilities, continuous learning, and the ability to adapt to new challenges. Ideal candidates "
        "will have a Bachelor's degree in Computer Science or a related field, 2+ years of experience in Python development, "
        "and familiarity with version control systems like Git. Knowledge of databases (SQL, NoSQL), RESTful APIs, and testing "
        "frameworks (pytest, unittest) is highly desirable. We offer a competitive salary, flexible work arrangements, and "
        "opportunities for professional growth in a fast-paced, innovative environment. If you are passionate about using "
        "technology to solve real-world problems and drive business success, we encourage you to apply and join our team "
        "of talented developers shaping the future of data-driven solutions."
    ),
    "Software Developer": (
        "Software Developer with experience building scalable applications using Python, Java, JavaScript, and related frameworks. "
        "Responsibilities include writing clean code, debugging, collaborating with teams, and deploying software. "
        "We seek developers who can design efficient algorithms, optimize performance, and ensure code quality through testing. "
        "You will work in agile environments, participate in code reviews, and integrate with CI/CD pipelines. "
        "Familiarity with databases, APIs, and version control is essential. Join our team to build innovative software solutions."
    ),
    "Software Engineer": (
        "Software Engineer responsible for designing, developing, and maintaining software systems using modern development practices. "
        "Involves coding, testing, version control, and ensuring software reliability. "
        "You will collaborate with product managers, designers, and other engineers to deliver high-quality products. "
        "Experience with object-oriented programming, design patterns, and software architecture is required. "
        "We value continuous improvement, learning new technologies, and contributing to open-source projects. "
        "Opportunities for career advancement in a supportive, inclusive workplace."
    ),
    "Software Architect": (
        "Software Architect who can design software structure, define technical standards, and guide engineering teams. "
        "Focuses on high-level design, scalability, and technology selection. "
        "You will evaluate emerging technologies, create architectural blueprints, and ensure alignment with business goals. "
        "Strong background in system design, microservices, and cloud architecture is needed. "
        "Leadership skills, communication abilities, and experience mentoring junior developers are key. "
        "Join us to shape the technical vision of our products."
    ),
    "Software Tester": (
        "Software Tester focused on automated and manual testing, QA processes, and bug tracking. "
        "Ensures software quality through unit, integration, and user acceptance testing. "
        "You will develop test plans, write test cases, and use tools like Selenium, JIRA, and Postman. "
        "Knowledge of testing methodologies, defect management, and performance testing is essential. "
        "We look for detail-oriented individuals who can work in fast-paced environments and improve product reliability."
    ),
    "Data Analyst": (
        "Data Analyst skilled in data visualization, statistical analysis, and tools like Excel, SQL, Python, and Tableau. "
        "Extracts insights from data, creates reports, and supports decision-making processes. "
        "You will clean and analyze large datasets, build dashboards, and present findings to stakeholders. "
        "Proficiency in statistical methods, data modeling, and business intelligence tools is required. "
        "Strong analytical skills and ability to translate data into actionable recommendations are valued."
    ),
    "Data Scientist": (
        "Data Scientist with expertise in machine learning, predictive modeling, and programming languages like Python and R. "
        "Builds models, analyzes large datasets, and communicates findings to drive business strategies. "
        "You will develop algorithms, perform A/B testing, and work with big data technologies. "
        "Knowledge of deep learning, NLP, and data visualization is essential. "
        "We seek curious, analytical minds who can solve complex problems with data."
    ),
    "Data Engineer": (
        "Data Engineer responsible for building data pipelines, ETL processes, and working with big data technologies like Hadoop and Spark. "
        "Ensures data integrity, performance, and accessibility for analytics. "
        "You will design scalable data architectures, automate data workflows, and optimize query performance. "
        "Experience with cloud platforms, streaming data, and database management is key. "
        "Join our team to enable data-driven decision-making."
    ),
    "Machine Learning Engineer": (
        "Machine Learning Engineer focused on developing ML models, deep learning, and deploying AI solutions. "
        "Works on algorithms, training models, and integrating ML into production systems. "
        "You will experiment with neural networks, handle model deployment, and monitor performance. "
        "Proficiency in TensorFlow, PyTorch, and MLOps is required. "
        "We value innovation and the ability to scale AI applications."
    ),
    "Business Analyst": (
        "Business Analyst who gathers requirements, analyzes business needs, and creates documentation for software projects. "
        "Bridges the gap between business and IT, ensuring solutions meet user needs. "
        "You will conduct stakeholder interviews, create use cases, and manage project scope. "
        "Skills in requirements elicitation, process modeling, and agile methodologies are essential. "
        "Strong communication and problem-solving abilities are key."
    ),
    "Web Developer": (
        "Web Developer proficient in HTML, CSS, JavaScript, and frameworks like React or Angular. "
        "Builds responsive websites, optimizes performance, and ensures cross-browser compatibility. "
        "You will create user interfaces, integrate APIs, and implement responsive design. "
        "Knowledge of web standards, accessibility, and version control is required. "
        "We seek creative developers who can deliver engaging web experiences."
    ),
    "Full Stack Developer": (
        "Full Stack Developer with skills in both frontend and backend technologies, including databases and APIs. "
        "Handles end-to-end development, from UI to server-side logic. "
        "You will build complete applications, manage databases, and ensure seamless integration. "
        "Proficiency in MERN/MEAN stack, RESTful services, and deployment is essential. "
        "Versatility and ability to work across the stack are valued."
    ),
    "Backend Developer": (
        "Backend Developer experienced in server-side logic, databases, and APIs using languages like Python or Node.js. "
        "Focuses on data processing, security, and API development. "
        "You will design scalable backends, implement authentication, and optimize performance. "
        "Knowledge of databases, caching, and microservices is key. "
        "We look for developers who can build robust server-side solutions."
    ),
    "Frontend Developer": (
        "Frontend Developer skilled in creating user interfaces with HTML, CSS, JavaScript, and libraries like React. "
        "Ensures intuitive and engaging user experiences. "
        "You will implement designs, handle state management, and optimize for mobile. "
        "Familiarity with UI/UX principles, testing, and build tools is required. "
        "Creativity and attention to detail are essential."
    ),
    "DevOps Engineer": (
        "DevOps Engineer responsible for CI/CD pipelines, automation, and infrastructure management. "
        "Improves deployment processes, monitoring, and system reliability. "
        "You will automate workflows, manage cloud infrastructure, and ensure security. "
        "Experience with tools like Jenkins, Terraform, and Kubernetes is key. "
        "We value efficiency and reliability in operations."
    ),
    "System Administrator": (
        "System Administrator managing servers, networks, and ensuring system reliability. "
        "Handles configuration, security, and troubleshooting. "
        "You will maintain hardware/software, monitor systems, and respond to incidents. "
        "Knowledge of Linux, networking, and virtualization is essential. "
        "Strong problem-solving skills in a technical environment."
    ),
    "Database Administrator": (
        "Database Administrator handling database design, optimization, and security. "
        "Manages data storage, backups, and performance tuning. "
        "You will ensure data availability, implement security measures, and optimize queries. "
        "Proficiency in SQL, NoSQL, and backup strategies is required. "
        "Attention to data integrity and performance."
    ),
    "UI/UX Designer": (
        "UI/UX Designer creating intuitive user interfaces and experiences. "
        "Conducts user research, wireframing, prototyping, and usability testing. "
        "You will design user flows, create mockups, and collaborate with developers. "
        "Skills in design tools like Figma, Adobe XD, and user-centered design are key. "
        "Creativity and empathy for users are valued."
    ),
    "Product Manager": (
        "Product Manager overseeing product development, roadmaps, and stakeholder communication. "
        "Defines product vision, prioritizes features, and analyzes market needs. "
        "You will gather requirements, manage backlogs, and track metrics. "
        "Experience in product lifecycle, data analysis, and cross-team collaboration is essential. "
        "Strategic thinking and leadership skills."
    ),
    "QA Engineer": (
        "QA Engineer ensuring software quality through testing and quality assurance processes. "
        "Develops test plans, automates tests, and reports defects. "
        "You will perform functional/non-functional testing, use automation tools, and improve QA processes. "
        "Knowledge of testing frameworks and bug tracking is required. "
        "Detail-oriented and committed to quality."
    ),
}

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def get_keywords(text):
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return [token for token in tokens if token not in ENGLISH_STOP_WORDS]

def top_keywords(text, n=10):
    tokens = get_keywords(text)
    counts = Counter(tokens)
    return counts.most_common(n)

def get_title_suggestions(prefix):
    query = prefix.strip().lower()
    if not query:
        return []
    starts = [title for title in JOB_TITLE_SUGGESTIONS if title.lower().startswith(query)]
    contains = [title for title in JOB_TITLE_SUGGESTIONS if query in title.lower() and title not in starts]
    return starts + contains

def format_keyword_list(words, max_items=30):
    return ", ".join(words[:max_items]) if words else "None"

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

job_title_input = st.text_input(
    "🔎 Type job title or keyword",
    placeholder="Try typing 'soft', 'data', or 'python'",
    help="Start typing a job title to get role suggestions and descriptions."
)

suggestions = get_title_suggestions(job_title_input)
if suggestions and job_title_input.strip():
    selected_suggestion = suggestions[0]
    st.session_state.job_description = JOB_DESCRIPTION_TEMPLATES.get(
        selected_suggestion, selected_suggestion
    )
else:
    selected_suggestion = ""

job_description = st.text_area(
    "💼 Paste Job Description",
    value=st.session_state.job_description,
    height=220,
    key="job_description",
)

use_tfidf = st.checkbox("Use TF-IDF similarity", value=False)

if uploaded_file is None:
    st.info("Upload your resume PDF first so the analyzer can compare it with the job description.")
elif not job_description.strip():
    st.warning("Enter a job description or type a job title above to auto-load one.")
else:
    resume_text = extract_text_from_pdf(uploaded_file)
    if not resume_text:
        st.error("Could not extract text from the uploaded PDF.")
    else:
        resume_keywords = sorted(set(get_keywords(resume_text)))
        jd_keywords = sorted(set(get_keywords(job_description)))

        if use_tfidf:
            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform([resume_text, job_description])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score = similarity * 100
            matched = []
            missing = []
        else:
            matched = [word for word in jd_keywords if word in resume_keywords]
            missing = [word for word in jd_keywords if word not in resume_keywords]
            score = (len(matched) / len(jd_keywords) * 100) if jd_keywords else 0.0

        score_value = min(max(int(score), 0), 100)

        st.subheader("📊 Match Score")
        st.progress(score_value)
        st.write(f"### {score:.2f}% match")

        metrics = st.columns(3)
        metrics[0].metric("Resume keywords", len(resume_keywords))
        metrics[1].metric("JD keywords", len(jd_keywords))
        metrics[2].metric("Method", "TF-IDF" if use_tfidf else "Keyword overlap")

        if not use_tfidf:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Matching keywords")
                st.write(format_keyword_list(matched, 50))
            with col2:
                st.subheader("❌ Missing keywords")
                st.write(format_keyword_list(missing, 50))

        st.subheader("📌 Top keywords")
        st.write(f"Resume: {format_keyword_list([f'{w} ({c})' for w, c in top_keywords(resume_text, 10)], 10)}")
        st.write(f"JD: {format_keyword_list([f'{w} ({c})' for w, c in top_keywords(job_description, 10)], 10)}")

        if st.checkbox("Show extracted resume text", key="show_resume_text"):
            st.text_area("Extracted Resume Text", resume_text, height=260)

        with st.expander("Resume sections extracted"):
            lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
            sections = {}
            current = None
            for line in lines:
                lower = line.lower()
                if lower.startswith(("skills", "experience", "education", "projects", "certifications", "summary", "profile")):
                    current = line
                    sections[current] = []
                    continue
                if current:
                    sections[current].append(line)
            if sections:
                for section, content in sections.items():
                    st.markdown(f"**{section}**")
                    st.write("\n".join(content))
            else:
                st.write("No recognizable sections found.")

        st.subheader("💡 Recommendations")
        if score > 85:
            st.success("Strong match! Your resume is well aligned.")
        elif score > 60:
            st.info("Good match, but you can still add a few keywords.")
        else:
            st.warning("Low match. Add more relevant skills, tools, and experience from the JD.")

        if not use_tfidf and missing:
            st.write("👉 Add these missing keywords:")
            st.write(format_keyword_list(missing, 30))

        report = (
            f"Resume analyzer report\n\n"
            f"Score: {score:.2f}%\n"
            f"Method: {'TF-IDF' if use_tfidf else 'Keyword overlap'}\n"
            f"Job title query: {job_title_input}\n"
            f"Selected role: {selected_suggestion}\n"
            f"Resume keywords: {len(resume_keywords)}\n"
            f"JD keywords: {len(jd_keywords)}\n"
            f"Matched keywords: {format_keyword_list(matched, 50)}\n"
            f"Missing keywords: {format_keyword_list(missing, 50)}\n\n"
            f"Top resume keywords:\n" + "\n".join(f"{w}: {c}" for w, c in top_keywords(resume_text, 10))
            + "\n\nTop JD keywords:\n" + "\n".join(f"{w}: {c}" for w, c in top_keywords(job_description, 10))
        )

        st.download_button(
            label="Download match report",
            data=report,
            file_name="resume_match_report.txt",
            mime="text/plain",
        )