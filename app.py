import os
from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import requests
import streamlit as st
import json

# Load environment variables
load_dotenv(".env")

AIML_API_KEY = os.getenv("AIML_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Create an AIML LLM instance
aiml_llm =LLM(
    model="gpt-4o",
    base_url="https://api.aimlapi.com/v1",
    api_key=AIML_API_KEY,
    temperature=0,
    max_tokens=1000
)

# AIML API Details
API_KEY = AIML_API_KEY  # Add your AIML API Key here
BASE_URL = "https://api.aimlapi.com/v1/stt"


import streamlit as st
import requests

# Function to send audio to AIML API for transcription with error handling
def transcribe_audio_with_aiml(audio_data):
    # Check if the audio file size is under 5MB
    if len(audio_data) > 5 * 1024 * 1024:  # 5 MB limit
        st.warning("⚠️ Please upload a smaller audio file (less than 5MB) for better performance.")
        return None

    url = BASE_URL
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    # Prepare the audio file for sending
    files = {"audio": ("audio.wav", audio_data, "audio/wav")}
    data = {"model": "#g1_whisper-large"}  # Model for transcription

    try:
        # Send audio data to AIML API with timeout set to 60 seconds
        response = requests.post(url, headers=headers, data=data, files=files, timeout=60)
        response.raise_for_status()  # Raise an error if the status code is 400 or higher
        
        # Parse the transcription result from the response
        response_data = response.json()
        transcript = response_data["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        return transcript

    except requests.exceptions.Timeout:
        st.warning("⏳ We are using AIML API for transcription as per hackathon guidelines. However, the service is taking too long to respond. Please try again later.")
        return None

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 524:
            st.warning("⚠️ We are using AIML API for transcription as per hackathon guidelines. Unfortunately, the service is currently unavailable (timeout). Please try again later.")
        else:
            st.warning(f"⚠️ We are using AIML API for transcription as per hackathon guidelines. An unexpected error occurred: {http_err}. Please try again later.")
        return None

    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ We are using AIML API for transcription as per hackathon guidelines. A network issue occurred: {e}. Please check your connection and try again.")
        return None

    except KeyError as e:
        st.warning("⚠️ We are using AIML API for transcription as per hackathon guidelines. The service returned an unexpected response. Please try again later.")
        return None



# Define Agents
input_collector = Agent(
    role="User Input Collector",
    goal="Gather and clarify user requirements for product search from text or voice input, with a focus on products available in Pakistan.",
    backstory="You are an expert in understanding user needs from various input types and translating them into clear search parameters, ensuring the results are relevant to users in Pakistan.",
    llm=aiml_llm,
    verbose=True
)

# Define  web search Tools
# Tools for specific websites
search_tool = SerperDevTool(api_key=SERPER_API_KEY)

scrape_google = ScrapeWebsiteTool(website_url='https://google.com/')
scrape_amazon = ScrapeWebsiteTool(website_url='https://www.amazon.com/')
scrape_daraz = ScrapeWebsiteTool(website_url='https://www.daraz.pk/')


web_searcher = Agent(
    role="Web Search Specialist",
    goal="Find product listings across Google, Amazon, and Daraz." \
    "Return the result as a JSON list of dictionaries, where each dictionary contains:"
        " 'name', 'price', 'rating', 'url', and 'image_url'.",
    backstory="You are a skilled product search expert who knows how to extract valuable listings from multiple platforms and format them properly.",
    tools=[search_tool, scrape_google, scrape_amazon, scrape_daraz],
    llm=aiml_llm,
    allow_delegation=False,
    verbose=True
)



analyst = Agent(
    role="Product Comparison Expert",
    goal="Evaluate the list of products based on quality, features, price, and reviews. Return the top 3 ranked recommendations with their name, price, rating, image URL, and purchase link.",
    backstory="You're an expert in product comparison. Based on the fetched listings, you compare and identify the top 3 options that offer the best value.",
    llm=aiml_llm,
    verbose=True
)

review_tool = WebsiteSearchTool(
    config={
        "llm": {
            "provider": "openai",  # LiteLLM-compatible wrapper
            "config": {
                "model": "gpt-4o",
                "api_key": AIML_API_KEY,
                "base_url": "https://api.aimlapi.com/v1",
                "temperature": 0.5,
                "max_tokens": 1000
            }
        },
        "embedder": {
            "provider": "openai",  # keep this for compatibility
            "config": {
                "model": "text-embedding-3-large",
                "api_key": AIML_API_KEY
            }
        }
    }
)

review_agent = Agent(
    role="Review Analyzer",
    goal="Analyze reviews and summarize user sentiment for the top product recommended by the analysis task from the specified vendor.",
    backstory="You analyze reviews from the selected vendor's website (Daraz, Amazon, AliExpress) for the chosen product to extract pros, cons, and overall user sentiment.",
    tools=[review_tool],
    llm=aiml_llm,
    verbose=True
)

# Define the final recommendation agent
recommender = Agent(
    role="Shopping Recommendation Specialist",
    goal="Present the top 3 product recommendations in an attractive format including product name, price, rating, image URL, and a purchase link. Keep the summary concise and helpful.",
    backstory="You help the user choose the best product by presenting the final comparison results. You summarize all relevant details and format it clearly for display.",
    llm=aiml_llm,
    verbose=True
)

# Define Tasks
filters = {
    "min_rating": 4.0,
    "brand": "Sony"
}

# Manually format the filters
brand = filters["brand"] if filters["brand"] else "Any"

description = (
    f"Process the user input: '{{user_input}}'\n"
    f"Use the following filters if applicable:\n"
    f"- Minimum Rating: {filters['min_rating']}\n"
    f"- Preferred Brand: {brand}\n"
    "Generate a refined product search query based on these inputs."
)

# Now, use the formatted description in your Task
input_task = Task(
    description=description,
    expected_output="A well-formed product search query based on the user's input and filter if any .",
    agent=input_collector
)


search_task = Task(
    description="""
        Search online for the best matching products using the refined search query.
        Look for product listings across Google, Amazon, and Daraz. Use appropriate tools for each platform 
        (e.g., Serper API for Google, scraping for Amazon and Daraz).

        Return a JSON list of the **top 3 products** with the following fields:
        - name (title)
        - price
        - rating
        - url
        - image_url
        - source (e.g., Amazon, Daraz, etc.)
    """,
    expected_output="""
        A JSON-formatted list of 3 product listings from Google, Amazon, and Daraz.
        Each product must include: name, price, rating, url, image_url, and source.
    """,
    agent=web_searcher,
    context=[input_task]
)


analysis_task = Task(
    description=(
        "Analyze the structured product listings (JSON format) from different websites (Google, Amazon, Daraz). "
        "Compare features, price, rating, and source for each. "
        "Rank the top 3 products based on overall value (considering quality, affordability, rating). "
        "For each of the top 3, return: name, price, rating, source, brief reason for ranking."
    ),
    expected_output="""
        A ranked list (1 to 3) of the top product recommendations.
        Each entry should include name, price, rating, source, and a short reason why it was selected.
    """,
    agent=analyst,
    context=[search_task]
)



review_task = Task(
    description=(
        "Using the top product recommendation and vendor from the analysis task, "
        "summarize customer reviews for this product. Review feedback will include pros, cons, and sentiment analysis. "
        "First, determine the correct website URL based on the vendor (Amazon, Daraz, or AliExpress). "
        "Then use the WebsiteSearchTool to analyze reviews from that specific website."
    ),
    expected_output="A summarized list of pros, cons, and user sentiment for the selected product from the appropriate vendor's website.",
    agent=review_agent,
    context=[analysis_task]
)

recommendation_task = Task(
    description=(
        "Provide a final product recommendation summary based on the top 3 ranked products and their customer reviews. "
        "Summarize key features, pros/cons, and customer sentiment for each. "
        "Highlight which one is the best choice and why, but present all three options with image URLs."
    ),
    expected_output="""
        A summary of top 3 recommended products.
        For each: name, price, rating, image_url, pros/cons, sentiment, and final verdict on the best one.
    """,
    agent=recommender,
    context=[
        analysis_task,
        review_task
    ]
)


product_knowledge = StringKnowledgeSource(
    content="Information about current product trends, including electronics, fashion, beauty, lifestyle, and more."
)

# Shopping Crew Setup
shopping_crew = Crew(
    agents=[input_collector, web_searcher, analyst, review_agent, recommender],
    tasks=[input_task, search_task, analysis_task, review_task, recommendation_task],
    verbose=True,
    process=Process.sequential,
    embedder={
        "provider": "aimlapi",
        "config": {
            "model": "text-embedding-3-large",
            "api_key": AIML_API_KEY,
            }
         }
    )



# --- Streamlit App UI ---
st.set_page_config(page_title="ShopSmart.AI", page_icon="🛒")



# Title and Logo in same row using columns
col1, col2 = st.columns([5, 2]) 
with col1:
    st.markdown("""
                <h1 style='margin-bottom: 0;'>🛍️ ShopSmart.AI</h1>
                <p style='margin-top: 0; font-size: 40px;'>Shop Smarter - Live Better</p>
                """, unsafe_allow_html=True)
with col2:
    st.image("tlogo.png", use_container_width=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Controls")
    
    # Handle reset from URL
    query_params = st.query_params
    if "reset" in query_params and query_params["reset"] == "1":
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    # Start new chat via query param
    if st.button("🧹 Start New Chat"):
        st.query_params["reset"] = "1"
        st.rerun()

    # Manual chat reset
    if st.button("🔄 Reset Chat"):
        st.session_state.clear()
        st.rerun()

    # Filters Section
    st.subheader("🔍 Filters (Optional)")

    filters = {
        "min_rating": st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.5),
        "brand": st.text_input("Preferred Brand", value="")
    }

    st.session_state["filters"] = filters
    st.write("Filters will be applied to the product search.")

# Filters are now stored in session state and ready to be used.
# --- Main Chat Area ---
st.markdown("<h5>💬 Just ask — your AI Shopping Crew will find, analyze, and deliver the best deals!</h5>", unsafe_allow_html=True)


# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Text"

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Display previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Mode Selector
input_mode = st.radio("Choose input type:", ("Text", "Voice"))
st.session_state.input_mode = input_mode

# Handle Text Input
if st.session_state.input_mode == "Text":
    user_input = st.chat_input("Type your query here...")
    if user_input:
        st.session_state.user_input = user_input

# Handle Voice Input
elif st.session_state.input_mode == "Voice":
    audio_data = st.audio_input("Speak to Record your Query")
    if audio_data:
        st.info("Processing audio...")
        transcribed_text = transcribe_audio_with_aiml(audio_data)
        if transcribed_text:
            st.session_state.user_input = transcribed_text

# --- Process after input is received ---
if st.session_state.user_input:
    user_msg = st.session_state.user_input
    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Assuming you have a function to process the user input, e.g., shopping_crew.kickoff()
            result = shopping_crew.kickoff(inputs={"user_input": user_msg})
            reply = result.raw

            try:
                # Try to parse JSON for structured product response
                products = json.loads(reply)
                if isinstance(products, list):
                    for product in products:
                        st.markdown(f"**🛍️ {product.get('title', 'No Title')}**")
                        st.image(product.get("image_url", ""), use_column_width=True)
                        st.markdown(f"""
                            💵 **Price:** {product.get('price', 'N/A')}  
                            ⭐ **Rating:** {product.get('rating', 'N/A')}  
                            📝 {product.get('description', 'No description')}
                            ---
                        """)
                    reply_summary = f"Found {len(products)} products. Please see above."
                else:
                    raise ValueError("Not a list")

            except Exception:
                # Fallback to plain text if JSON fails
                st.markdown(reply)
                reply_summary = reply

        # Save assistant message to session history (summary or plain)
        st.session_state.messages.append({"role": "assistant", "content": reply_summary})

    # Clear input after processing
    st.session_state.user_input = ""

# Footer

st.markdown("""
    <style>
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: gray;
        background-color: white;
        z-index: 100;
    }
    .custom-footer hr {
        border: none;
        border-top: 1px solid #ddd;
        margin: 0;
    }
    </style>

    <div class="custom-footer">
        <hr>
        Powered by Streamlit | Developed by The Team TechPulse
    </div>
    """, unsafe_allow_html=True)

