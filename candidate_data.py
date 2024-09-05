def get_candidate_data():
    candidates = [
        {
            "rank": 3,  # Updated rank
            "candidate": "Divisha_Jain_DA.pdf",
            "match_score": 60,
            "recommendation": "Interview with reservations",
            "brief_summary": "Divisha Jain has a strong background in data analytics with some experience in machine learning techniques, but lacks specific knowledge in Generative AI, Deep Learning systems, and reinforcement learning.",
            "experience_relevance": "Divisha's experience in delivering insights via advanced analytics and machine learning techniques for various domains is relevant. However, she lacks specific industrial application of ML, Generative AI, and Deep Learning systems.",
            "skills_gap": [
                "Generative AI",
                "Deep Learning Systems",
                "Reinforcement Learning",
                "Scalable Machine Learning"
            ],
            "recruiter_questions": [
                "Can you elaborate on your experience with machine learning techniques?",
                "Do you have any experience with Generative AI or Deep Learning systems?",
                "Do you have any knowledge or experience in reinforcement learning?",
                "Can you discuss any projects where you had to scale your Machine Learning models?"
            ],
            "project_relevance": "No specific projects relevant to large-scale machine learning systems."
        },
        {
            "rank": 4,  # Updated rank
            "candidate": "Artem Nyzhnyk_SDE.pdf",
            "match_score": 32,
            "recommendation": "Do not recommend for interview",
            "brief_summary": "Artem Nyzhnyk, a 20+ year experienced software engineer, has a good background in full-stack development, technical leadership, and solution architecture, but lacks direct experience with machine learning and AI.",
            "experience_relevance": "While Artem has a strong background in software engineering, his skills and experience do not match the job requirements. He has some relevant experience in data analysis and visualization, but it is not directly applicable to this role.",
            "skills_gap": [
                "Machine learning and deep learning",
                "Generative AI",
                "Probability and statistics",
                "Linear algebra",
                "Python",
                "TensorFlow",
                "pyTorch",
                "langchain",
                "Scikit-learn"
            ],
            "recruiter_questions": [
                "Can you explain the concept of deep learning and how it differs from traditional machine learning?",
                "How do you approach feature engineering for a machine learning model?",
                "Can you describe a time when you had to handle large-scale data and develop a strategy for efficient data processing?",
                "How do you stay up-to-date with the latest developments in machine learning and AI?"
            ],
            "project_relevance": "Artem's personal projects, such as the Tower Defense Game, are not relevant to this role as they do not showcase machine learning or AI expertise. However, his experience in developing a photo album with semantic search powered by AI is somewhat relevant."
        },
        # More candidates can be added here if needed
    ]
    return candidates
