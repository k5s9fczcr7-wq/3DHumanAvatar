"""The avatar's name: kiosk_config.ini [avatar] name (default Kiah). Her prompt and greetings use it."""
import configparser
import os

DEFAULT_NAME = "Kiah"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kiosk_config.ini")


def load_name(path: str = CONFIG_PATH) -> str:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path, encoding="utf-8")
    except configparser.Error:
        return DEFAULT_NAME
    return cfg.get("avatar", "name", fallback="").strip() or DEFAULT_NAME


AVATAR_NAME = load_name()
