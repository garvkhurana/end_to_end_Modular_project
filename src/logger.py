import logging
import os
from datetime import datetime

log_file = datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".log"
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, log_file)

logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == '__main__':
    logging.info('This is a test log message')
