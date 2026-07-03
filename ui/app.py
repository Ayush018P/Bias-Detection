import streamlit as st
import sys
from pathlib import Path

# Fix sys path FIRST before importing local modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from db.database import engine, Base, SessionLocal
from db.models import User
# Initialize Database tables
Base.metadata.create_all(bind=engine)

st.set_page_config(page_title="SABPF Login", layout="centered")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

def login():
    st.title("Welcome to SABPF Enterprise")
    st.markdown("Please log in or create an account to access the bias detection framework.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            
            if submitted:
                db = SessionLocal()
                user = db.query(User).filter(User.username == username).first()
                if user and user.verify_password(password):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user.username
                    st.session_state["user_id"] = user.id
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                db.close()
                
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            signup_submitted = st.form_submit_button("Create Account")
            
            if signup_submitted:
                if len(new_username) < 3 or len(new_password) < 5:
                    st.warning("Username must be >= 3 chars, password >= 5 chars.")
                else:
                    db = SessionLocal()
                    existing = db.query(User).filter(User.username == new_username).first()
                    if existing:
                        st.error("Username already taken.")
                    else:
                        hashed_pw = User.get_password_hash(new_password)
                        new_user = User(username=new_username, hashed_password=hashed_pw)
                        db.add(new_user)
                        db.commit()
                        st.success("Account created! You can now log in.")
                    db.close()

if not st.session_state["authenticated"]:
    login()
else:
    st.title(f"Welcome back, {st.session_state['username']}!")
    st.markdown("You are authenticated. Please select a page from the sidebar to begin your bias audit workflows.")
    
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state["user_id"] = None
        st.rerun()

    # Clear memory of datasets when on home page
    st.info("👈 Navigate to the **Dataset Explorer** to load data.")
