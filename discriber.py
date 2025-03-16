import os
import openai
import inspect
import time
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

def get_code_description(code):
    """Get a description of the code using OpenAI's GPT-4o-mini model."""
    try:
        # Create a client using the new API pattern
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Use streaming API with optimized parameters
        response_text = ""
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Describe code in exactly 2 sentences: what it does and what functions it interacts with."},
                {"role": "user", "content": f"Code:\n{code}"}
            ],
            stream=True,
            temperature=0.2,
            timeout=5  # 5 second timeout
        )
        
        # Process the streaming response
        print("Receiving description...")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response_text += content
                print(content, end="", flush=True)
        print("\n")
        
        return response_text
    except Exception as e:
        return f"Error getting description: {str(e)}"

def main():
    # Wait for "yes" input
    user_input = input("Type 'yes' to get descriptions of all functions: ").strip().lower()
    
    if user_input != "yes":
        print("Operation cancelled. Please type 'yes' to proceed.")
        return
    
    # Start the timer
    start_time = time.time()
    
    try:
        # Import all functions from test.py
        import test
        
        # Get all functions from the test module
        functions = [obj for name, obj in inspect.getmembers(test) 
                    if inspect.isfunction(obj) and obj.__module__ == test.__name__]
        
        # Process each function
        for i, func in enumerate(functions, 1):
            func_name = func.__name__
            print(f"\n{'-'*50}")
            print(f"Function {i} of {len(functions)}: {func_name}")
            
            # Get the source code of the function
            func_code = inspect.getsource(func)
            
            # Get the description
            func_description = get_code_description(func_code)
            print(f"Description: {func_description}")
        
        # End the timer and display elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n{'-'*50}")
        print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    
    except ImportError:
        print("Error: Could not import functions from test.py.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()