import os

# Default to 'development' if ENV_TYPE is not set
ENV_TYPE = os.getenv('ENV_TYPE', 'development')

class Config:
    if ENV_TYPE == 'production':
        LOG_DIR = "/home/rezcupid2024/Resume_Cupid_Multi_LLM/logs"
        CHROME_BINARY = "/usr/bin/chromium-browser"
        DB_PATH = "/home/rezcupid2024/Resume_Cupid_Multi_LLM/resume_cupid.db"
    else:
        LOG_DIR = "./logs"
        CHROME_BINARY = None
        DB_PATH = "./resume_cupid.db"

    @staticmethod
    def get_chrome_options():
        from selenium.webdriver.chrome.options import Options
        options = Options()
        if Config.CHROME_BINARY:
            options.binary_location = Config.CHROME_BINARY
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-extensions")
        options.add_argument("--window-size=1920x1080")
        return options