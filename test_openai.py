# Add this test script to verify your OpenAI API key
# Save as test_openai.py and run it separately

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai_key():
    """Test OpenAI API key and show detailed information"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    print("=" * 60)
    print("OpenAI API Key Test")
    print("=" * 60)
    
    # Check if key exists
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("\nPlease add to .env:")
        print("OPENAI_API_KEY=sk-proj-your-key-here")
        return False
    
    # Check key format
    print(f"✓ Key found: {OPENAI_API_KEY[:20]}...{OPENAI_API_KEY[-10:]}")
    print(f"✓ Key length: {len(OPENAI_API_KEY)} characters")
    
    # Try to initialize client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client initialized")
    except ImportError:
        print("❌ ERROR: openai package not installed")
        print("Run: pip install openai>=1.45.0")
        return False
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize client: {e}")
        return False
    
    # Test API call with minimal tokens
    try:
        print("\n" + "=" * 60)
        print("Testing API call...")
        print("=" * 60)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'test' only"}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✓ API call successful!")
        print(f"✓ Response: {result}")
        print(f"✓ Model: {response.model}")
        
        # Show usage
        if hasattr(response, 'usage'):
            print(f"✓ Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
        # Parse error for specific issues
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            print("\n💡 QUOTA ERROR DETECTED:")
            print("   1. Check your OpenAI billing: https://platform.openai.com/account/billing")
            print("   2. Verify payment method is valid")
            print("   3. Check if you have any spending limits set")
            print("   4. Free tier has very low limits - consider upgrading")
        elif "401" in error_str or "authentication" in error_str.lower():
            print("\n💡 AUTHENTICATION ERROR:")
            print("   1. Your API key might be invalid or revoked")
            print("   2. Generate a new key: https://platform.openai.com/api-keys")
        elif "404" in error_str:
            print("\n💡 MODEL NOT FOUND:")
            print("   1. You might not have access to gpt-4o-mini")
            print("   2. Try 'gpt-3.5-turbo' instead")
        
        return False

if __name__ == "__main__":
    success = test_openai_key()
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED - Your OpenAI setup is working!")
    else:
        print("❌ TESTS FAILED - Please fix the issues above")
    print("=" * 60)