import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VRHateSpeechGenerator:
    def __init__(self):
        self.openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.base_prompt = """Generate a realistic conversation that might occur 
        in VR voice chat. 
The conversation should:
- Include 2 users having a casual conversation
- Include subtle toxic/hate speech elements, such as insults, 
  threats, sexism, racism, and other forms of hate speech
- Focus on verbal interaction only (like voice chat, in any VR context.)
- The conversation should cover all topics: include general topics such 
  as movies, music, daily chatting, etc.
- Be realistic and conversational
- The conversation can be just a few rounds (2-3 rounds) of back-and-forth 
  messages
- Show both toxic and non-toxic participants
- Format as: [Username]: message

The toxic behavior should be realistic, similar to what might occur in 
real VR voice chats."""

    def generate_conversation(self):
        try:
            completion = self.openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are simulating realistic VR voice chat conversations that include toxic behavior."
                    },
                    {
                        "role": "user",
                        "content": self.base_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return self.parse_conversation(completion.choices[0].message.content)
        except Exception as error:
            print(f'Error generating conversation: {error}')
            raise error
    
    def parse_conversation(self, response):
        messages = []
        for line in response.split('\n'):
            if line.startswith('['):
                try:
                    username_part, message = line.split(']: ', 1)
                    username = username_part.replace('[', '').strip()
                    messages.append({
                        "username": username,
                        "message": message.strip(),
                        "timestamp": datetime.now().isoformat()
                    })
                except ValueError:
                    # Skip lines that don't match the expected format
                    continue
        
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "message_count": len(messages),
                "participants": list(set(m["username"] for m in messages))
            },
            "messages": messages
        }
    
    def generate_dataset(self, count=3):
        dataset = []
        
        for i in range(count):
            try:
                conversation = self.generate_conversation()
                dataset.append(conversation)
                print(f"Generated conversation {i + 1}/{count}")
            except Exception as error:
                print(f"Error generating conversation {i}: {error}")
        
        return dataset


if __name__ == "__main__":
    # Generate conversations
    generator = VRHateSpeechGenerator()
    
    try:
        dataset = generator.generate_dataset(2)
        
        # Save to file
        with open('vr_hate_conversations_dataset.json', 'a') as f:
            json.dump(dataset, f, indent=2)
        
        print('Dataset saved to vr_hate_conversations_dataset.json')
    except Exception as error:
        print(f'Error: {error}')