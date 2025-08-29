import requests
import json

# Base URL
BASE_URL = "http://127.0.0.1:8000/api"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat(message="Tell me about your background"):
    """Test the chat endpoint"""
    try:
        payload = {
            "message": message,
            "session_id": "test_session_123"
        }
        
        response = requests.post(
            f"{BASE_URL}/chat", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nChat Test Status: {response.status_code}")
        print(f"Request: {json.dumps(payload, indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"Error Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("=" * 50)
    print("TESTING BACKEND API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Endpoint...")
    health_ok = test_health()
    
    # Test 2: Chat Endpoint
    print("\n2. Testing Chat Endpoint...")
    chat_ok = test_chat("What programming languages do you know?")
    
    # Test 3: Another chat message
    print("\n3. Testing Another Chat Message...")
    chat_ok2 = test_chat("Tell me about your projects")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Chat Endpoint: {'✅ PASS' if chat_ok else '❌ FAIL'}")
    print(f"Multiple Messages: {'✅ PASS' if chat_ok2 else '❌ FAIL'}")
    
    if health_ok and chat_ok and chat_ok2:
        print("\n🎉 ALL TESTS PASSED! Your backend is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    run_all_tests()