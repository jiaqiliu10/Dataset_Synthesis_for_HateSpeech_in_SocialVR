import os
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VRConversationGenerator:
    def __init__(self):
        self.openai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.base_prompt = """Generate a realistic conversation that might occur 
    in VR voice chat. 
The conversation should:
- Include 2-3 users having a casual, friendly conversation
- Cover diverse topics such as:
  * Movies, TV shows, and entertainment
  * Music, concerts, and artists
  * Daily life, work, and hobbies
  * Travel and experiences
  * Food and cooking
  * Current events and culture
  * Technology and VR experiences
- Show positive social interactions like:
  * Natural small talk and friendly banter
  * Sharing recommendations and opinions
  * Finding common interests
- Be realistic and natural
- Include 1-2 rounds of back-and-forth messages
- Format as: [Username]: message"""

    def generate_conversation(self):
        try:
            completion = self.openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are simulating realistic, positive VR voice chat conversations."
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
    generator = VRConversationGenerator()
    
    try:
        dataset = generator.generate_dataset(2)
        
        # Save to file
        with open('vr_non_hate_conversations_dataset.json', 'a') as f:
            json.dump(dataset, f, indent=2)
        
        print('Dataset saved to vr_non_hate_conversations_dataset.json')
    except Exception as error:
        print(f'Error: {error}')