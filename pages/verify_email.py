import streamlit as st
from database import verify_user_email

def main():
    st.title("Email Verification")

    # Get query parameters
    query_params = st.query_params
    st.write("Debug - Query Params:", query_params)  # Debugging line
    
    token = query_params.get('token', None)
    st.write("Debug - Token:", token)  # Debugging line

    if token:
        success = verify_user_email(token)
        if success:
            st.success("Your email has been successfully verified! You can now log in.")
        else:
            st.error("The verification link is invalid or expired.")
    else:
        st.error("No verification token provided.")

if __name__ == "__main__":
    main()