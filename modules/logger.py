import logging

logger = logging.getLogger("ComfyUI Cleaner")
logger.setLevel(logging.INFO)
logger.propagate = False

logger_sh = logging.StreamHandler()
logger_sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger_sh.setLevel(logging.INFO)
logger.addHandler(logger_sh)
