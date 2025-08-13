# # app.py
# import streamlit as st
# import pandas as pd
# from utils.file_utils import extract_text_from_pdf, extract_text_from_docx, convert_docx_to_pdf
# from utils.nlp_utils import compute_tfidf_similarity, compute_bert_similarity
# from utils.normalization import normalize_scores
# import numpy as np

# # Streamlit UI
# st.set_page_config(page_title="AI Resume Screening", layout="wide")
# st.title("AI Resume Screening & Ranking System")

# # Session state
# if "resume_texts" not in st.session_state:
#     st.session_state.resume_texts = {}

# # Sidebar
# with st.sidebar:
#     st.image("images/shutterstock_546995980.webp")
#     st.header("AI Resume Screening & Ranking")
#     st.write(" ")

#     if st.button("Search Resumes"):
#         st.session_state["show_search"] = True
#         st.session_state["show_about"] = False

#     if st.button("About"):
#         st.session_state["show_about"] = True
#         st.session_state["show_search"] = False

#     if st.button("Download Results") and "results_df" in st.session_state:
#         df = st.session_state["results_df"]
#         if not df.empty:
#             csv = df.to_csv(index=False).encode('utf-8')
#             st.download_button("Download Ranked Resumes", csv, "ranked_resumes.csv", "text/csv")
#         else:
#             st.warning("No results to download.")

# # Main functionality
# jd_text = st.text_area("Enter Job Description:").strip()
# uploaded_resumes = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

# if jd_text and uploaded_resumes:
#     resume_texts = {}
#     for resume in uploaded_resumes:
#         if resume.name in st.session_state.resume_texts:
#             text = st.session_state.resume_texts[resume.name]
#         else:
#             text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_docx(resume)
#             st.session_state.resume_texts[resume.name] = text if text.strip() else "N/A"
#         if text.strip() != "N/A":
#             resume_texts[resume.name] = text

#     if resume_texts:
#         names = list(resume_texts.keys())
#         texts = list(resume_texts.values())
#         tfidf_scores = normalize_scores(np.array([compute_tfidf_similarity(jd_text, t) for t in texts]))
#         bert_scores = normalize_scores(np.array([compute_bert_similarity(jd_text, t) for t in texts]))
#         results_df = pd.DataFrame({
#             "Resume Name": names,
#             "TF-IDF Score": tfidf_scores,
#             "BERT Score": bert_scores
#         })
#         results_df["Final Score"] = (results_df["TF-IDF Score"] + results_df["BERT Score"]) / 2
#         results_df = results_df.sort_values(by="Final Score", ascending=False)
#         st.session_state["results_df"] = results_df
#         st.write("### Ranked Resumes")
#         st.dataframe(results_df)
# import streamlit as st
# import pandas as pd
# import numpy as np
# from utils.file_utils import extract_text_from_pdf, extract_text_from_docx, convert_docx_to_pdf
# from utils.nlp_utils import compute_tfidf_similarity, compute_bert_similarity
# from utils.normalization import normalize_scores

# # ------------------ PAGE CONFIG ------------------
# st.set_page_config(
#     page_title="AI Resume Screening & Ranking",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ------------------ SESSION STATE INIT ------------------
# def reset_state():
#     st.session_state.resume_texts = {}
#     st.session_state.results_df = pd.DataFrame()
#     st.session_state.page = "Home"

# if "resume_texts" not in st.session_state:
#     st.session_state.resume_texts = {}
# if "results_df" not in st.session_state:
#     st.session_state.results_df = pd.DataFrame()
# if "page" not in st.session_state:
#     st.session_state.page = "Home"

# # ------------------ SIDEBAR ------------------
# with st.sidebar:
#     st.image("images/shutterstock_546995980.webp", width=180)
#     st.title("AI Resume Screening")

#     # Navigation
#     st.session_state.page = st.radio(
#         "Navigation",
#         ["Home", "Search Resumes", "About"],
#         index=["Home", "Search Resumes", "About"].index(st.session_state.page)
#     )

#     st.divider()

#     # Download Ranked CSV
#     if not st.session_state.results_df.empty:
#         csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
#         st.download_button("📥 Download Ranked CSV", csv, "ranked_resumes.csv", "text/csv")

#     # Clear / Reload
#     if st.button("🔄 Clear / Reload"):
#         reset_state()
#         st.rerun()

# # ------------------ HOME PAGE ------------------
# if st.session_state.page == "Home":
#     st.title("AI Resume Screening & Ranking System")
#     st.markdown("Upload multiple resumes (PDF/DOCX) and input a job description to get ranked resumes.")

#     jd_text = st.text_area("Enter Job Description:").strip()
#     uploaded_resumes = st.file_uploader(
#         "Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True
#     )

#     ranking_method = st.radio(
#         "Select Ranking Method:",
#         ["Combined (TF-IDF + BERT)", "TF-IDF Only", "BERT Only"]
#     )

#     if jd_text and uploaded_resumes:
#         resume_texts = {}
#         progress_text = st.empty()
#         progress_bar = st.progress(0)

#         # Process Resumes
#         for i, resume in enumerate(uploaded_resumes, 1):
#             progress_text.text(f"Processing {i}/{len(uploaded_resumes)}: {resume.name}")
#             progress_bar.progress(i / len(uploaded_resumes))

#             if resume.name in st.session_state.resume_texts:
#                 text = st.session_state.resume_texts[resume.name]
#             else:
#                 text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_docx(resume)
#                 st.session_state.resume_texts[resume.name] = text if text.strip() else "N/A"

#             if text.strip() != "N/A":
#                 resume_texts[resume.name] = text

#         progress_text.empty()
#         progress_bar.empty()

#         if resume_texts:
#             names = list(resume_texts.keys())
#             texts = list(resume_texts.values())

#             tfidf_scores = normalize_scores(np.array([compute_tfidf_similarity(jd_text, t) for t in texts]))
#             bert_scores = normalize_scores(np.array([compute_bert_similarity(jd_text, t) for t in texts]))

#             results_df = pd.DataFrame({
#                 "Resume Name": names,
#                 "TF-IDF Score": tfidf_scores,
#                 "BERT Score": bert_scores
#             })

#             if ranking_method == "TF-IDF Only":
#                 results_df["Final Score"] = tfidf_scores
#             elif ranking_method == "BERT Only":
#                 results_df["Final Score"] = bert_scores
#             else:
#                 results_df["Final Score"] = (tfidf_scores + bert_scores) / 2

#             results_df = results_df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
#             st.session_state.results_df = results_df

#             # Display Results
#             st.subheader("Ranked Resumes")
#             st.dataframe(results_df)

#             st.subheader("Top 5 Resumes")
#             st.bar_chart(results_df.head(5).set_index("Resume Name")["Final Score"])

#             st.subheader("Download Individual Resumes")
#             for name in results_df["Resume Name"]:
#                 content = resume_texts[name]
#                 if name.endswith(".docx"):
#                     pdf_bytes = convert_docx_to_pdf(content)
#                     st.download_button(f"📥 {name} as PDF", pdf_bytes, name.replace(".docx", ".pdf"), "application/pdf")
#                 else:
#                     st.download_button(f"📥 {name}", content, name, "application/pdf")

# # ------------------ SEARCH PAGE ------------------
# elif st.session_state.page == "Search Resumes":
#     st.subheader("Search Resumes")
#     search_query = st.text_input("Enter resume name or keyword:")
#     if search_query and not st.session_state.results_df.empty:
#         df = st.session_state.results_df
#         matched = df[df["Resume Name"].str.contains(search_query, case=False, na=False)]
#         if not matched.empty:
#             st.dataframe(matched)
#         else:
#             st.warning("No matches found.")
#     elif st.session_state.results_df.empty:
#         st.warning("No resumes processed yet.")

# # ------------------ ABOUT PAGE ------------------
# elif st.session_state.page == "About":
#     st.subheader("About This System")
#     st.markdown("""
#     This AI-powered resume screening system ranks resumes based on their relevance to a job description using **TF-IDF** and **BERT similarity**.

#     **Features:**
#     - Extracts text from **PDF and DOCX resumes**, including image-based PDFs via OCR
#     - Computes similarity scores with **TF-IDF & BERT**
#     - Multiple ranking methods: **Combined / TF-IDF / BERT**
#     - Visualizes **Top-N Resumes**
#     - Search resumes by keyword
#     - Download ranked results as CSV
#     - Convert DOCX to PDF for download
#     - Clear / Reload session data
#     - Ready for **skill extraction** and shortlist features
#     """)
# import streamlit as st
# import pandas as pd
# import numpy as np

# from utils.file_utils import extract_text_from_pdf, extract_text_from_docx, convert_docx_to_pdf
# from utils.nlp_utils import compute_tfidf_similarity, compute_bert_similarity, extract_skills
# from utils.normalization import normalize_scores

# # ------------------ PAGE CONFIG ------------------
# st.set_page_config(
#     page_title="AI Resume Screening & Ranking",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ------------------ SESSION STATE INIT ------------------
# def reset_state():
#     st.session_state.resume_texts = {}
#     st.session_state.results_df = pd.DataFrame()
#     st.session_state.resume_skills = {}
#     st.session_state.page = "Home"

# if "resume_texts" not in st.session_state:
#     st.session_state.resume_texts = {}
# if "results_df" not in st.session_state:
#     st.session_state.results_df = pd.DataFrame()
# if "resume_skills" not in st.session_state:
#     st.session_state.resume_skills = {}
# if "page" not in st.session_state:
#     st.session_state.page = "Home"

# # ------------------ SIDEBAR ------------------
# with st.sidebar:
#     st.image("images/shutterstock_546995980.webp", width=180)
#     st.title("AI Resume Screening")

#     st.session_state.page = st.radio(
#         "Navigation",
#         ["Home", "Search Resumes", "About"],
#         index=["Home", "Search Resumes", "About"].index(st.session_state.page)
#     )

#     st.divider()
#     if not st.session_state.results_df.empty:
#         csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
#         st.download_button("📥 Download Ranked CSV", csv, "ranked_resumes.csv", "text/csv")

#     if st.button("🔄 Clear / Reload"):
#         reset_state()
#         st.rerun()

# # ------------------ HOME PAGE ------------------
# if st.session_state.page == "Home":
#     st.title("AI Resume Screening & Ranking System")
#     st.markdown(
#         "Upload multiple resumes (PDF/DOCX), input a job description, and get ranked results with skill match scores."
#     )

#     jd_text = st.text_area("Enter Job Description:").strip()
#     uploaded_resumes = st.file_uploader(
#         "Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True
#     )

#     ranking_method = st.radio(
#         "Select Ranking Method:", ["Combined (TF-IDF + BERT)", "TF-IDF Only", "BERT Only"]
#     )

#     if jd_text and uploaded_resumes:
#         resume_texts = {}
#         resume_skills = {}
#         progress_text = st.empty()
#         progress_bar = st.progress(0)

#         # Extract skills from job description
#         jd_skills = extract_skills(jd_text)

#         for i, resume in enumerate(uploaded_resumes, 1):
#             progress_text.text(f"Processing {i}/{len(uploaded_resumes)}: {resume.name}")
#             progress_bar.progress(i / len(uploaded_resumes))

#             if resume.name in st.session_state.resume_texts:
#                 text = st.session_state.resume_texts[resume.name]
#             else:
#                 if resume.name.endswith(".pdf"):
#                     text = extract_text_from_pdf(resume)
#                 else:
#                     text = extract_text_from_docx(resume)
#                 st.session_state.resume_texts[resume.name] = text if text.strip() else "N/A"

#             if text.strip() != "N/A":
#                 resume_texts[resume.name] = text
#                 skills = extract_skills(text)
#                 resume_skills[resume.name] = skills
#                 st.session_state.resume_skills[resume.name] = skills

#         progress_text.empty()
#         progress_bar.empty()

#         if resume_texts:
#             names = list(resume_texts.keys())
#             texts = list(resume_texts.values())

#             # ----- Compute Similarity Scores -----
#             tfidf_scores = normalize_scores(
#                 np.array([compute_tfidf_similarity(jd_text, t) for t in texts])
#             )
#             bert_scores = normalize_scores(
#                 np.array([compute_bert_similarity(jd_text, t) for t in texts])
#             )

#             # ----- Create Results DataFrame -----
#             results_df = pd.DataFrame({
#                 "Resume Name": names,
#                 "TF-IDF Score": tfidf_scores,
#                 "BERT Score": bert_scores,
#                 "Skill Match Score": [
#                     len(set(resume_skills[n]).intersection(set(jd_skills))) / max(len(jd_skills), 1) * 100
#                     for n in names
#                 ]
#             })

#             # ----- Final Scoring -----
#             if ranking_method == "TF-IDF Only":
#                 results_df["Final Score"] = (
#                     tfidf_scores * 0.7 + results_df["Skill Match Score"] * 0.3
#                 )
#             elif ranking_method == "BERT Only":
#                 results_df["Final Score"] = (
#                     bert_scores * 0.7 + results_df["Skill Match Score"] * 0.3
#                 )
#             else:
#                 combined = (tfidf_scores + bert_scores) / 2
#                 results_df["Final Score"] = (
#                     combined * 0.7 + results_df["Skill Match Score"] * 0.3
#                 )

#             results_df = results_df.sort_values(
#                 by="Final Score", ascending=False
#             ).reset_index(drop=True)
#             st.session_state.results_df = results_df

#             # ----- Display Results -----
#             st.subheader("Ranked Resumes")
#             st.dataframe(results_df)

#             st.subheader("Top 5 Resumes")
#             st.bar_chart(results_df.head(5).set_index("Resume Name")["Final Score"])

#             st.subheader("Download Individual Resumes")
#             for name in results_df["Resume Name"]:
#                 content = resume_texts[name]
#                 if name.endswith(".docx"):
#                     pdf_bytes = convert_docx_to_pdf(content)
#                     st.download_button(
#                         f"📥 {name} as PDF", pdf_bytes,
#                         name.replace(".docx", ".pdf"), "application/pdf"
#                     )
#                 else:
#                     st.download_button(
#                         f"📥 {name}", content, name, "application/pdf"
#                     )

# # ------------------ SEARCH PAGE ------------------
# elif st.session_state.page == "Search Resumes":
#     st.subheader("Search Resumes")
#     search_query = st.text_input("Enter resume name or keyword:")
#     if search_query and not st.session_state.results_df.empty:
#         df = st.session_state.results_df
#         matched = df[df["Resume Name"].str.contains(search_query, case=False, na=False)]
#         if not matched.empty:
#             st.dataframe(matched)
#         else:
#             st.warning("No matches found.")
#     elif st.session_state.results_df.empty:
#         st.warning("No resumes processed yet.")

# # ------------------ ABOUT PAGE ------------------
# elif st.session_state.page == "About":
#     st.subheader("About This System")
#     st.markdown("""
#     This AI-powered resume screening system ranks resumes based on their relevance to a job description 
#     using **TF-IDF** and **BERT similarity**, with added **Skill Match scoring**.

#     **Features:**
#     - Extracts text from **PDF and DOCX resumes**, including image-based PDFs via OCR.
#     - Computes similarity scores using **TF-IDF & BERT**.
#     - Provides **skill match percentage** based on JD skills.
#     - Multiple ranking methods: Combined / TF-IDF / BERT.
#     - Visualizes **Top-N Resumes**.
#     - Search & download functionality for results and individual resumes.
#     - Option to **clear/reload session data**.
#     - Converts **DOCX resumes to PDF** for convenient downloads.
#     """)
"""
app.py
AI Resume Screening & Ranking System
------------------------------------
- Processes PDF/DOCX resumes (with OCR for scanned PDFs)
- Ranks resumes using TF-IDF, BERT, and hybrid scoring
- Extracts skills (categorized) + named entities
- Search, download, and visualize results
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.file_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    convert_docx_to_pdf
)
from utils.nlp_utils import (
    compute_tfidf_similarity,
    compute_bert_similarity,
    extract_skills,
    extract_skills_with_categories,
    skill_match_score,
    extract_entities
)
from utils.normalization import normalize_scores

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Resume Screening & Ranking",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ SESSION STATE INIT ------------------
DEFAULT_SESSION = {
    "resume_texts": {},
    "results_df": pd.DataFrame(),
    "resume_skills": {},
    "resume_entities": {},
    "page": "Home"
}

for k, v in DEFAULT_SESSION.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_state():
    """Reset the session state for a fresh start."""
    for k, v in DEFAULT_SESSION.items():
        st.session_state[k] = v


# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.image("images/shutterstock_546995980.webp", width=180)
    st.title("AI Resume Screening")

    st.session_state.page = st.radio(
        "Navigation",
        ["Home", "Search Resumes", "About"],
        index=["Home", "Search Resumes", "About"].index(st.session_state.page)
    )

    st.divider()

    if not st.session_state.results_df.empty:
        csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Ranked CSV",
            csv,
            "ranked_resumes.csv",
            "text/csv"
        )

    if st.button("🔄 Clear / Reload"):
        reset_state()
        st.rerun()


# ------------------ HOME PAGE ------------------
if st.session_state.page == "Home":
    st.title("AI Resume Screening & Ranking System")
    st.markdown(
        """
        Upload multiple resumes (PDF/DOCX), enter a job description, and get ranked results with skill & entity insights.
        """
    )

    jd_text = st.text_area("📄 Job Description").strip()

    uploaded_resumes = st.file_uploader(
        "📂 Upload Resumes (PDF/DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    ranking_method = st.radio(
        "⚖ Select Ranking Method:",
        ["Combined (TF-IDF + BERT)", "TF-IDF Only", "BERT Only"]
    )

    if jd_text and uploaded_resumes:
        resume_texts, resume_skills, resume_entities = {}, {}, {}

        progress_text = st.empty()
        progress_bar = st.progress(0)

        jd_skills = extract_skills(jd_text)  # Pre-extract JD skills

        for i, resume in enumerate(uploaded_resumes, 1):
            progress_text.text(f"Processing {i}/{len(uploaded_resumes)}: {resume.name}")
            progress_bar.progress(i / len(uploaded_resumes))

            text = st.session_state.resume_texts.get(resume.name)
            if not text:
                try:
                    if resume.name.lower().endswith(".pdf"):
                        text = extract_text_from_pdf(resume)
                    else:
                        text = extract_text_from_docx(resume)
                except Exception as e:
                    text = f"Error reading file: {e}"
                st.session_state.resume_texts[resume.name] = text

            if text and text.strip() and not text.startswith("Error"):
                resume_texts[resume.name] = text
                skills = extract_skills_with_categories(text)
                entities = extract_entities(text)

                resume_skills[resume.name] = skills
                resume_entities[resume.name] = entities

                st.session_state.resume_skills[resume.name] = skills
                st.session_state.resume_entities[resume.name] = entities

        progress_text.empty()
        progress_bar.empty()

        if resume_texts:
            names, texts = list(resume_texts.keys()), list(resume_texts.values())

            # ----- Compute Scores -----
            tfidf_scores = normalize_scores(
                np.array([compute_tfidf_similarity(jd_text, t) for t in texts])
            )
            bert_scores = normalize_scores(
                np.array([compute_bert_similarity(jd_text, t) for t in texts])
            )
            skill_scores = np.array([
                skill_match_score(resume_texts[n], jd_text) * 100 for n in names
            ])

            results_df = pd.DataFrame({
                "Resume Name": names,
                "TF-IDF Score": tfidf_scores,
                "BERT Score": bert_scores,
                "Skill Match %": skill_scores
            })

            # ----- Final Scoring -----
            if ranking_method == "TF-IDF Only":
                results_df["Final Score"] = tfidf_scores * 0.7 + skill_scores * 0.3
            elif ranking_method == "BERT Only":
                results_df["Final Score"] = bert_scores * 0.7 + skill_scores * 0.3
            else:
                combined_scores = (tfidf_scores + bert_scores) / 2
                results_df["Final Score"] = combined_scores * 0.7 + skill_scores * 0.3

            results_df = results_df.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
            st.session_state.results_df = results_df

            # ----- Display Results -----
            st.subheader("📊 Ranked Resumes")
            st.dataframe(results_df)

            st.subheader("🏆 Top 5 Resumes")
            st.bar_chart(results_df.head(5).set_index("Resume Name")["Final Score"])

            # ----- Detailed Resume Insights -----
            st.subheader("🔍 Resume Details")
            for name in results_df["Resume Name"]:
                with st.expander(f"Details for {name}"):
                    st.markdown("**Skills by Category:**")
                    st.json(resume_skills.get(name, {}))

                    st.markdown("**Extracted Entities:**")
                    st.json(resume_entities.get(name, {}))

            # ----- Download Resumes -----
            st.subheader("📥 Download Individual Resumes")
            for name in results_df["Resume Name"]:
                content = resume_texts[name]
                if name.lower().endswith(".docx"):
                    pdf_bytes = convert_docx_to_pdf(content)
                    st.download_button(
                        f"{name} as PDF",
                        pdf_bytes,
                        name.replace(".docx", ".pdf"),
                        "application/pdf"
                    )
                else:
                    st.download_button(
                        name,
                        content,
                        name,
                        "application/pdf"
                    )


# ------------------ SEARCH PAGE ------------------
elif st.session_state.page == "Search Resumes":
    st.subheader("🔎 Search Resumes")
    search_query = st.text_input("Enter resume name or keyword:")

    if search_query and not st.session_state.results_df.empty:
        df = st.session_state.results_df
        matched = df[df["Resume Name"].str.contains(search_query, case=False, na=False)]
        if not matched.empty:
            st.dataframe(matched)
        else:
            st.warning("No matches found.")
    elif st.session_state.results_df.empty:
        st.warning("No resumes processed yet.")


# ------------------ ABOUT PAGE ------------------
elif st.session_state.page == "About":
    st.subheader("ℹ About This System")
    st.markdown("""
    This AI-powered resume screening system ranks resumes for relevance to a job description 
    using **TF-IDF** and **BERT** similarity, plus **Skill Match Scoring** from a curated skills database.

    **Features:**
    - 📂 Supports **PDF/DOCX** resumes, including scanned PDFs (OCR).
    - 🧠 Hybrid ranking: TF-IDF + BERT + skill match.
    - 🗂 Skill extraction with **category grouping** from CSV.
    - 🏷 Named entity recognition for education, companies, and more.
    - 🔎 Search & download ranked results.
    - 📥 Individual resume conversion to PDF.
    - ♻ Clear/reload for fresh sessions.
    """)
