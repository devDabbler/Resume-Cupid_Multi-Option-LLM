import streamlit as st
from database import verify_user_email  # Assuming this function verifies the token and updates the database

def main():
    # Get the token from the URL parameters
    query_params = st.query_params
    st.write("Query Params:", query_params)  # Debugging line to check the query params
    token = query_params.get('token', [None])[0]

    # Check if the token is valid
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
