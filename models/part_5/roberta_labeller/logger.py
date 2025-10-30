import logging
import sys

# ==== LOGGER INFO SETUP ====
_log_info = logging.getLogger("info_logger")
_log_info.setLevel(logging.INFO)

_handler = logging.StreamHandler(sys.stdout)  # explicitly write to stdout
_formatter = logging.Formatter("[%(asctime)s] - [%(levelname)s] - %(message)s")
_handler.setFormatter(_formatter)
_log_info.addHandler(_handler)

# ==== LOGGER DIVIDER SETUP ====
_log_divider = logging.getLogger("divider")
_log_divider.setLevel(logging.WARN)
_divider_handler = logging.StreamHandler()
_divider_handler.setFormatter(logging.Formatter("%(message)s"))
_log_divider.addHandler(_divider_handler)

def divider_logger():
    _log_divider.info("=" * 80)
    
def info_logger(message):
    _log_info.info(message)