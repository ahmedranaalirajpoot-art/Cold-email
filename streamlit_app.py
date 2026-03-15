import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(page_title="Cold Email Generator", page_icon="📧")
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1f2937);
    color: white;
}

/* Input fields */
.stTextInput>div>div>input {
    background-color: #1e293b;
    color: white;
    border-radius: 8px;
    border: 1px solid #3b82f6;
}

.stSelectbox>div>div {
    background-color: #1e293b;
    color: white;
    border-radius: 8px;
    border: 1px solid #3b82f6;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: white;
}

/* Text area */
textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Headers */
h1, h2, h3 {
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

st.title("📧 Cold Email Generator")
st.markdown("Generate and send personalized cold emails using AI")

with st.sidebar:
    st.header("Configuration")
    
    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )
    
    sender_email = st.text_input(
        "Your Gmail",
        value="ahmedranaalirajpoot@gmail.com"
    )
    
    sender_password = st.text_input(
        "Gmail App Password",
        type="password"
    )

st.header("Enter Details")

website_url = st.text_input("Target Website URL")
recipient_email = st.text_input("Recipient Email")

service_name = st.selectbox(
    "Service to Offer",
    ["SEO Optimization", "Website Repair", "Content Writing", "Social Media Marketing", "Web Design"]
)

col1, col2 = st.columns(2)

with col1:
    sender_name = st.text_input("Your Name")

with col2:
    sender_company = st.text_input("Your Company")


def generate_email(website, service, name, company, api_key):
    
    scraping_tool = ScrapeWebsiteTool()
    llm = LLM(model="gemini/gemini-2.5-flash", api_key=api_key)
    
    scraper = Agent(
        role="Web Researcher",
        goal=f"Research {website}",
        backstory="Expert website analyst",
        tools=[scraping_tool],
        llm=llm
    )
    
    email_writer = Agent(
        role="Email Copywriter",
        goal=f"Write cold email offering {service}",
        backstory="Professional outreach copywriter",
        llm=llm
    )
    
    sender_agent = Agent(
        role="Email Sender",
        goal="Format final email professionally",
        backstory="Email formatting expert",
        llm=llm
    )
    
    research_task = Task(
        description=f"Analyze {website} and summarize the business.",
        expected_output="Business summary",
        agent=scraper
    )
    
    email_task = Task(
        description=f"""
Write a cold email.

Research: {{research_task.output}}
Sender: {name} from {company}
Service: {service}

Keep under 150 words.
""",
        expected_output="Email draft",
        agent=email_writer
    )
    
    sender_task = Task(
        description=f"""
Finalize the email with signature.

Draft: {{email_task.output}}
Name: {name}
Company: {company}
""",
        expected_output="Final formatted email",
        agent=sender_agent
    )
    
    crew = Crew(
        agents=[scraper, email_writer, sender_agent],
        tasks=[research_task, email_task, sender_task],
        verbose=True
    )
    
    result = crew.kickoff()
    
    return result.raw  # ✅ IMPORTANT FIX


def send_email_smtp(to_email, subject, body, sender_email, sender_password):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))  # body is now string ✅
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


if st.button("Generate & Send Email", type="primary", use_container_width=True):
    
    if not api_key:
        st.error("Enter Gemini API key")
    elif not website_url or not recipient_email or not sender_name or not sender_company:
        st.error("Please fill all fields")
    else:
        with st.spinner("Generating email..."):
            try:
                email_text = generate_email(
                    website_url,
                    service_name,
                    sender_name,
                    sender_company,
                    api_key
                )
                
                st.success("Email generated!")
                
                st.subheader("Generated Email:")
                st.text_area("", value=email_text, height=250)
                
                with st.spinner("Sending email..."):
                    subject = f"{service_name} Services"
                    
                    success = send_email_smtp(
                        recipient_email,
                        subject,
                        email_text,  # ✅ Now string
                        sender_email,
                        sender_password.replace(" ", "")
                    )
                    
                    if success:
                        st.success("✅ Email sent successfully!")
                    else:
                        st.error("❌ Failed to send email. Check credentials.")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")