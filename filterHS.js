import fs from 'fs';
import path from 'path';

const inputDir = './json_files';
const outputFile = 'filtered_conversations.json';

async function filterAndSaveConversations() {
  try {
    const files = await fs.promises.readdir(inputDir);
    let allConversations = [];

    for (let file of files) {
      if (path.extname(file) === '.json') {
        const filePath = path.join(inputDir, file);
        const data = JSON.parse(await fs.promises.readFile(filePath, 'utf8'));
        const filteredData = data.filter(convo => convo.messages && convo.messages.length > 0);
        allConversations = allConversations.concat(filteredData);
      }
    }

    // Write the filtered conversations to a new file
    await fs.promises.writeFile(outputFile, JSON.stringify(allConversations, null, 2));
    console.log(`Filtered conversations saved to ${outputFile}`);
    console.log(`Total number of conversations: ${allConversations.length}`);
  } catch (error) {
    console.error('Error filtering conversations:', error);
  }
}

// Run the function
filterAndSaveConversations();
