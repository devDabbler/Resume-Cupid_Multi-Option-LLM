# Resume Cupid 💘

Resume Cupid is an intelligent resume evaluation tool designed to streamline the hiring process using advanced AI models. It analyzes and ranks resumes against job descriptions, providing detailed insights to help recruiters and hiring managers make informed decisions.

**Note: This project is in early stages of development and is not yet ready for public use or contributions.**

## Project Overview

Resume Cupid aims to revolutionize the hiring process by leveraging cutting-edge AI technology to match candidates with job openings more effectively. Our tool provides a comprehensive analysis of resumes, helping recruiters and hiring managers save time and make data-driven decisions.

## Current Features and Future Plans

- **AI-Powered Analysis**: Currently utilizing Llama 8B Instant for resume analysis. Future updates will allow users to choose their preferred Language Model (LLM).
- **Resume Processing**: Extraction of text from PDF and DOCX files
- **Job Description Matching**: Analysis of resumes against specific job requirements
- **Customizable Importance Factors**: Adjustable weights for education, experience, and skills
- **Stack Ranking**: Automatic ranking of candidates based on their match scores
- **Detailed Analysis**: In-depth analysis of each resume, including strengths, areas for improvement, and skills gaps
- **Interview Questions**: Generation of tailored interview questions for each candidate
- **Feedback Collection**: Gathering of user feedback for continuous improvement
- **Role Management**: Saving and managing job descriptions for future use
- **User Authentication**: Secure login system for authorized access
- **Performance Optimization**: Parallel processing of resumes for faster results

## Project Status

Resume Cupid is currently in active development. We are working on refining our AI model integration, improving the user interface, and ensuring the highest standards of data privacy and security. Future updates will include the ability for users to integrate their LLM of choice.

## Installation

To set up Resume Cupid for local development:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/resume-cupid.git
   cd resume-cupid
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add necessary API keys and configurations.

5. Run the Streamlit app:
   ```
   streamlit run main_app.py
   ```

## Usage

(To be added once the project reaches a more stable state. This section will include basic usage examples and screenshots.)

## Future Plans

As we progress, we plan to:
1. Implement flexibility for users to choose and integrate their preferred LLM
2. Conduct extensive testing to ensure accuracy and reliability
3. Develop a robust API for potential integrations
4. Explore partnerships with HR software providers
5. Enhance our AI models with more specialized industry knowledge

## Contributing

We are not currently accepting external contributions as the project is in its early stages. However, we appreciate your interest and will update this section when we're ready for community contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries about Resume Cupid or potential collaborations, please contact us at hello@resumecupid.ai.

## Acknowledgements

We would like to acknowledge the following technologies and organizations that are instrumental in the development of Resume Cupid:

- [Meta AI](https://ai.facebook.com/) for the Llama model
- [Streamlit](https://streamlit.io/) for the web application framework
- [Spacy](https://spacy.io/) for natural language processing
- [PyPDF2](https://pypdf2.readthedocs.io/) and [python-docx](https://python-docx.readthedocs.io/) for document parsing

---

© 2024 Resume Cupid AI. All rights reserved.