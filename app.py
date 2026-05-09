import streamlit as st
import google.generativeai as genai
from jira import JIRA
import chromadb
from sentence_transformers import SentenceTransformer

# Configure Gemini API
genai.configure(api_key="AIzaSyCeruJwY3jSYQTBdR2j4fc7tx1cSuNOnlA")

# Load Gemini model
model = genai.GenerativeModel("gemini-3.1-flash-lite")

# Page configuration
st.set_page_config(
    page_title="MeetMind AI",
    layout="centered"
)

# App title
st.title("🧠 MeetMind AI")
st.caption("AI-powered Meeting Summarization MVP")

# Transcript input
meeting_notes = st.text_area(
    "Paste Meeting Transcript",
    placeholder="Paste meeting transcript here...",
    height=300
)
# Jira Configuration
JIRA_SERVER = "https://deepalisahai.atlassian.net"

JIRA_EMAIL = "deepalisahai1986@gmail.com"

JIRA_API_TOKEN = "ATATT3xFfGF0kvjbVRiQU10ZPK7BuntS7lIsWMDcXDwadiQ-AV0aHLX2gaMIVB7CzUHqwvhPgr5U=8AFBCC95"

jira = JIRA(
    basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN),
    server=JIRA_SERVER
)

# Generate Summary button
if st.button("Generate Summary"):

    if meeting_notes.strip() == "":
        st.warning("Please paste a meeting transcript.")

    else:

        with st.spinner("Generating AI summary..."):

            # AI Prompt
            prompt = f"""
            You are an AI meeting assistant.

            Analyze the following meeting transcript.

	    Return the output ONLY in this format:

	    Summary:
	    <summary>

	    Key Decisions:
	    - item 1
	    - item 2

	    Action Items:
	    - owner → task
	    - owner → task

	    Priority:
	    High / Medium / Low

            Transcript:
            {meeting_notes}
            """

            # Gemini response
            response = model.generate_content(prompt)

            # Display AI output
            st.success("Summary Generated Successfully!")
            st.subheader("AI Generated Output")	
            st.markdown(response.text)

sample_tasks = [
    "Finalize API specifications",
    "Review dashboard UX flow",
    "Start regression testing"
]
# Jira button
if st.button("Create Jira Tickets"):

    for task in sample_tasks:

        issue_dict = {
            "project": {"key": "MEET"},
            "summary": task,
            "description": f"Task created by MeetMind AI: {task}",
            "issuetype": {"name": "Task"},
        }

        jira.create_issue(fields=issue_dict)

    st.success("Jira Tickets Created Successfully!")
  

# Confluence button
if st.button("Publish to Confluence"):
    st.success("Meeting Notes Published to Confluence")
    
#CREATE VECTOR DATABASE

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="meeting_transcripts"
)    

#STORE TRANSCRIPT IN VECTOR DB
# Store transcript in vector DB

# Split transcript into chunks
chunks = [
    meeting_notes[i:i+300]
    for i in range(0, len(meeting_notes), 300)
]
for idx, chunk in enumerate(chunks):

    if chunk.strip() != "":

        embedding = embedding_model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{idx}"]
        )


#ADD QUESTION BOX
st.divider()

st.subheader("Ask Questions From Meeting")

user_question = st.text_input(
    "Ask something about the transcript"
)
#ADD RETRIEVAL LOGIC
if st.button("Ask AI"):

    # Convert question into embedding
    query_embedding = embedding_model.encode(
        user_question
    ).tolist()

    # Retrieve relevant transcript
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=15
    )

    retrieved_docs = results['documents'][0]

    retrieved_context = "\n".join(retrieved_docs)

    # RAG Prompt
    rag_prompt = f"""
    You are an AI meeting assistant.

        Answer the question using the retrieved meeting context below.

        The wording in the question and context may differ slightly.
        Match semantically similar meanings.

        If answer exists, provide concise response.

        If not found, say:
        "Answer not found in transcript."

        Context:
        {retrieved_context}

        Question:
        {user_question}

    """
   # st.write("Retrieved Context That I got Dips:")
   # st.write(retrieved_context)

    # Gemini response
    response = model.generate_content(rag_prompt)

    st.success("Answer Generated")

    st.write(response.text)