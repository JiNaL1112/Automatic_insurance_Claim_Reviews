import logging
import json
import sys
from api.config import settings

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "time": self.formatTime(record),
        })

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    if not logger.handlers:
        logger.addHandler(handler)
    return logger