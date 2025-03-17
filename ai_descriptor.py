import os
import openai
import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIDescriptor:
    """
    AI Descriptor class to generate descriptions for code elements using OpenAI API.
    """

    def __init__(self, api_key=None):
        """
        Initializes the AIDescriptor with an OpenAI API key.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided - AI description generation will not work")
        else:
            openai.api_key = self.api_key
            logger.info("AIDescriptor initialized with API key")
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key)
            self.headers = {"Authorization": f"Bearer {self.api_key}"}
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o-mini"  # Or another model of your choice
            self.description_cache = {} # Initialize cache


    async def describe_code_element(self, code_snippet, element_type, focus_area: Optional[str] = None) -> str:
        """
        Generates a description for a given code element using OpenAI API asynchronously.

        Args:
            code_snippet (str): The code snippet to describe.
            element_type (str): The type of code element (function, class, method).
            focus_area (Optional[str]): Specific area to focus the description on.

        Returns:
            Optional[str]: A description of the code element, or None if generation fails.
        """
        cache_key = (code_snippet, element_type, focus_area)
        if cache_key in self.description_cache: # Check cache first
            logger.info(f"Using cached description for {element_type}")
            return self.description_cache[cache_key]

        # Create system prompt based on element type and focus area
        system_prompt = f"Describe this {element_type} in 1-2 sentences:"
        if focus_area:
            system_prompt = f"Describe this {element_type} in 1-2 sentences, focusing on its role in {focus_area}:"

        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Code:\n{code_snippet}"}
        ]

        try:
            response = await self.async_client.chat.completions.create( # Use async client
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=150
            )

            description = response.choices[0].message.content.strip()
            logger.info(f"Generated description for {element_type} ({len(description)} chars)")
            self.description_cache[cache_key] = description # Cache the description
            return description

        except openai.APIError as e:
            logger.error(f"OpenAI API error generating description: {e}")
            return f"OpenAI API error: {str(e)}"
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Error generating description: {str(e)}"

    def get_confirmed_description(self, code: str, element_type: str,
                                 element_name: str, focus_area: Optional[str] = None) -> str:
        """
        Generate a description and get user confirmation.

        Args:
            code: Source code to describe
            element_type: Type of code element
            element_name: Name of the element
            focus_area: Optional functional path to focus on

        Returns:
            Confirmed description
        """
        # Generate initial description
        description = asyncio.run(self.describe_code_element(code, element_type, focus_area))

        # Format display text
        if focus_area:
            print(f"\nGenerated description for {element_type} '{element_name}' with focus on {focus_area}:")
        else:
            print(f"\nGenerated description for {element_type} '{element_name}':")

        print("-" * 50)
        print(description)
        print("-" * 50)

        # Get confirmation
        print("Press Enter to confirm this description or type modifications:")
        user_input = input().strip()

        # Return original if confirmed, otherwise use the modified text
        return description if not user_input else user_input 