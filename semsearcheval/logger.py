import logging


def setup_logger(name: str, log_level=logging.INFO, log_to_file=False, log_file_path="app.log"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(module)s.%(funcName)s l.%(lineno)d: %(message)s"
        )

        # Create console handler and add to logger
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Optional: Add file handler if specified
        if log_to_file:
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # Suppress HuggingFace and SentenceTransformers logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    return logger


logger = setup_logger(__name__)
