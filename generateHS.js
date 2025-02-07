import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config();

class VRHateSpeechGenerator {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    this.basePrompt = `Generate a realistic conversation that might occur 
    in VR voice chat. 
The conversation should:
- Include 2 users having a casual conversation
- Include very toxic/hate speech elements, such as insults, 
  threats, sexism, racism, and other forms of hate speech
- Focus on verbal interaction only (like voice chat, in any VR context.)
- The conversation should cover all topics: include general topics such 
  as movies, music, daily chatting, etc.
- Be realistic and conversational
- The conversation can be just a few rounds (2-3 rounds) of back-and-forth 
  messages
- Show both toxic and non-toxic participants
- Format as: [Username]: message

The toxic behavior should be realistic, similar to what might occur i
n real VR voice chats.`;
  }

  async generateConversation() {
    try {
      const completion = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [{
          role: "system",
          content: "You are simulating realistic VR voice chat conversations that include toxic behavior."
        }, {
          role: "user",
          content: this.basePrompt
        }],
        temperature: 0.7,
        max_tokens: 500
      });

      return this.parseConversation(completion.choices[0].message.content);
    } catch (error) {
      console.error('Error generating conversation:', error);
      throw error;
    }
  }

  parseConversation(response) {
    const messages = response.split('\n')
      .filter(line => line.startsWith('['))
      .map(line => {
        const [username, message] = line.split(']: ');
        return {
          username: username.replace('[', '').trim(),
          message: message.trim(),
          timestamp: new Date()
        };
      });

    return {
      metadata: {
        generated_at: new Date(),
        message_count: messages.length,
        participants: [...new Set(messages.map(m => m.username))]
      },
      messages
    };
  }

  async generateDataset(count = 3) {
    const dataset = [];
    
    for (let i = 0; i < count; i++) {
      try {
        const conversation = await this.generateConversation();
        dataset.push(conversation);
        console.log(`Generated conversation ${i + 1}/${count}`);
      } catch (error) {
        console.error(`Error generating conversation ${i}:`, error);
      }
    }

    return dataset;
  }
}

// Generate conversations
const generator = new VRHateSpeechGenerator();

generator.generateDataset(100)
  .then(dataset => {
    // Save to file
    import('fs').then(fs => {
      fs.appendFileSync('vr_conversations_dataset.json', JSON.stringify(dataset, null, 2));
      console.log('Dataset saved to vr_conversations_dataset.json');
    });
  })
  .catch(error => {
    console.error('Error:', error);
  });