FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libnss3 \
    libx11-6 \
    libxrender1 \
    libfontconfig1 \
    libxtst6 \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install spaCy English language model
RUN python -m spacy download en_core_web_md

# Install ChromeDriver
RUN curl -sS -o /usr/local/bin/chromedriver https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip \
    && chmod +x /usr/local/bin/chromedriver

# Install Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
