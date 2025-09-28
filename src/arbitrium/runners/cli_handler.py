from typing import Any

import colorama

from arbitrium.interfaces.event_handler import EventHandler
from arbitrium.logging import get_contextual_logger
from arbitrium.utils.display import Display


class ConsoleEventHandler(EventHandler):
    def __init__(self) -> None:
        colorama.init(autoreset=True)
        # Logging is initialized in run_from_cli() before this handler is created
        self.logger = get_contextual_logger("arbitrium.cli_handler")
        self.display = Display()

    def publish(self, event_name: str, data: dict[str, Any]) -> None:
        if event_name == "log":
            level = data.get("level", "info").lower()
            message = data.get("message", "")
            if hasattr(self.logger, level):
                getattr(self.logger, level)(message)
        elif event_name == "display":
            display_type = data.get("type")
            if display_type == "section_header":
                self.display.print_section_header(data.get("text", ""))
            elif display_type == "print":
                color = data.get("color") or ""
                self.display.print(data.get("text", ""), level_or_color=color)
            elif display_type == "model_response":
                self.display.print_model_response(data.get("model", ""), data.get("content", ""))
            elif display_type == "header":
                self.display.header(data.get("text", ""))
