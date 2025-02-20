import OpenAI from 'openai';
import dotenv from 'dotenv';
import fs from 'fs/promises';

dotenv.config();

class VRNonHateSpeechGenerator {
  constructor() {
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    this.basePrompt = `Generate a realistic conversation that might occur in VR voice chat. 
The conversation should:
- Include 2 users having a casual and friendly conversation
- Be natural, engaging, and respectful
- Focus on verbal interaction only (like voice chat)
- Be positive, inclusive, and non-toxic
- Format as: [Username]: message

Below are some examples:

Example 1:
[User1]: Hey! Your avatar looks awesome. Where did you get that skin?
[User2]: Thanks! I got it from the VR marketplace last week. You should check it out, they have some cool designs.
[User1]: Nice! I've been thinking about getting a new one. Any recommendations?
[User2]: Totally! I'll send you a link after this session.

Example 2:
[User1]: Wow, this world is amazing! Have you been here before?
[User2]: Yeah, I love this place! The lighting effects are so realistic.
[User1]: Agreed! It almost feels like we're inside a movie scene.
[User2]: Exactly! That's why I love VR—feels like stepping into another universe.

Example 3:
[User1]: Hey, do you play any VR games outside of chat?
[User2]: Yeah! I'm into VR racing and rhythm games. What about you?
[User1]: Oh, nice! I love rhythm games too. Ever tried that new VR dance battle one?
[User2]: Not yet, but I've heard good things. Maybe we can play together sometime!
`;
  }

  async generateConversation() {
    try {
      const completion = await this.openai.chat.completions.create({
        model: "gpt-4",
        messages: [{
          role: "system",
          content: "You are simulating friendly and engaging VR voice chat conversations."
        }, {
          role: "user",
          content: this.basePrompt
        }],
        temperature: 0.7,
        max_tokens: 800
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

  async generateDataset(count = 40) {
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

  async saveDatasetToFile(dataset, filename = 'vr_non_hate_speech_dataset_100.json') {
    try {
      let existingData = [];

      // Read existing data from the file if it exists
      try {
        const fileContent = await fs.readFile(filename, 'utf8');
        existingData = JSON.parse(fileContent);
      } catch (err) {
        if (err.code !== 'ENOENT') throw err; // Ignore error if file doesn't exist
      }

      // Append new data to existing data
      existingData.push(...dataset);

      // Write back to the file
      await fs.writeFile(filename, JSON.stringify(existingData, null, 2));
      console.log(`Dataset successfully appended to ${filename}`);
    } catch (error) {
      console.error('Error saving dataset:', error);
    }
  }
}

// Generate conversations
const generator = new VRNonHateSpeechGenerator();

// Generate a smaller dataset first for testing
generator.generateDataset(100)
  .then(dataset => generator.saveDatasetToFile(dataset))
  .catch(error => console.error('Error:', error))
  
//node non_hate.js