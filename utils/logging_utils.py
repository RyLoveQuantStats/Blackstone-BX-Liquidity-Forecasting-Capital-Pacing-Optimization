import logging
import os

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename='logs/database_operations.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)

if __name__ == "__main__":
    setup_logging()
    log_info("Logging setup completed. Logs will be stored in the 'logs' folder.")
