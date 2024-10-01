import streamlit as st
from database import reset_user_password

def main():
    st.title("Reset Password")

    # Get query parameters
    query_params = st.query_params
    st.write("Debug - Query Params:", query_params)  # Debugging line
    
    token = query_params.get('token', None)
    st.write("Debug - Token:", token)  # Debugging line
    
    if token:
        new_password = st.text_input("Enter your new password", type="password")
        confirm_password = st.text_input("Confirm your new password", type="password")
        
        if st.button("Reset Password"):
            if new_password == confirm_password:
                success = reset_user_password(token, new_password)
                if success:
                    st.success("Your password has been successfully reset! You can now log in.")
                else:
                    st.error("The reset link is invalid or expired.")
            else:
                st.error("Passwords do not match.")
    else:
        st.error("No token provided in the URL.")

if __name__ == "__main__":
    main()