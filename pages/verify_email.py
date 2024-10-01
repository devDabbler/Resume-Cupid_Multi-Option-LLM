import streamlit as st
from database import verify_user_email  # Assuming this is where the verification function resides

def main():
    # Extract token from URL parameters
    query_params = st.query_params
    token = query_params.get('token', [None])[0]

    if token:
        # Verify the token and update the database
        if verify_user_email(token):
            st.success("Email verified successfully! You can now log in.")
        else:
            st.error("Invalid or expired verification link.")
    else:
        st.error("No verification token provided.")

if __name__ == "__main__":
    main()
