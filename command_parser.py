import re
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CommandParser:
    """Parses text commands for CodeMemory AI system."""
    
    def __init__(self):
        """Initialize the command parser with command patterns."""
        # Command patterns with regex
        self.command_patterns = {
            'add_file': r'(?:add|import|include)\s+([a-zA-Z0-9_./-]+\.[\w]+)',
            'describe': r'(?:describe|explain|what\s+is)\s+([a-zA-Z0-9_./-]+)',
            'search': r'(?:search|find|look\s+for)\s+(.*?)$',
            'help': r'(?:help|commands|what\s+can\s+you\s+do)',
            'quit': r'(?:quit|exit|bye)'
        }
        logger.info("CommandParser initialized with %d command patterns", len(self.command_patterns))
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text input into a structured command.
        
        Args:
            text: User input text
            
        Returns:
            Dictionary containing command type and parameters
        """
        text = text.strip().lower()
        logger.debug("Parsing command: %s", text)
        
        for cmd_type, pattern in self.command_patterns.items():
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                logger.info("Matched command type: %s", cmd_type)
                
                if cmd_type == 'add_file':
                    return {
                        'command': cmd_type,
                        'file_path': match.group(1)
                    }
                elif cmd_type == 'describe':
                    return {
                        'command': cmd_type,
                        'element_id': match.group(1)
                    }
                elif cmd_type == 'search':
                    return {
                        'command': cmd_type,
                        'query': match.group(1)
                    }
                else:
                    return {'command': cmd_type}
        
        logger.warning("Unknown command: %s", text)
        return {'command': 'unknown', 'text': text}
    
    def get_help_text(self) -> str:
        """Return help text with available commands."""
        help_text = "Available commands:\n"
        help_text += "  - add <file_path>: Add a file to the project\n"
        help_text += "  - describe <element_id>: Get description of a code element\n"
        help_text += "  - search <query>: Search for code elements\n"
        help_text += "  - help: Show this help text\n"
        help_text += "  - quit: Exit the program"
        return help_text

def confirm_action(prompt: str = "Proceed? (Press Enter to confirm or type 'cancel'): ") -> bool:
    """Get confirmation from user before proceeding with an action."""
    response = input(prompt).strip().lower()
    return response == "" or response not in ["cancel", "no", "n"] 