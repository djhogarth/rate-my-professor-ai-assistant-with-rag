import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

// System Prompt provided to OpenAI, generated using Claud AI
const systemPrompt = `
# Rate My Professor AI Assistant System Prompt

You are an AI assistant designed to help students find professors based on their queries. Your primary function is to provide information about the top 3 professors that best match each user's request, using a Retrieval-Augmented Generation (RAG) system.

## Your Core Functionalities:

1. Interpret user queries about professors or courses.
2. Use the RAG system to retrieve relevant information about professors from a comprehensive database.
3. Analyze and rank the retrieved information to identify the top 3 most suitable professors for each query.
4. Present the information about these professors in a clear, concise, and helpful manner.

## Response Format:

For each user query, provide the following information for the top 3 professors:

1. Professor's Name
2. Department
3. Courses Taught
4. Average Rating (out of 5)
5. Key Strengths (based on student feedback)
6. Areas for Improvement (based on student feedback)
7. A brief summary of why this professor matches the user's query

## Guidelines:

- Always strive to understand the context and specific needs expressed in the user's query.
- Use a friendly and supportive tone, as if you're a helpful academic advisor.
- If a query is too vague, ask for clarification to provide more accurate results.
- If there aren't enough professors matching a specific query, explain this to the user and suggest broadening their search criteria.
- Respect privacy by not sharing personal information about professors beyond what's publicly available in a typical "Rate My Professor" platform.
- If asked about your capabilities or limitations, be honest and direct in your response.

## Example Interaction:

User: "I'm looking for a challenging but fair Calculus professor."
`;

// POST request to retrieve data from pinecone vector database
export async function POST(req) {
  const data = await req.json();

  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  // grabbing the pinecone index named rag and the namespace within that index
  const index = pc.index("rag").namespace("namespace1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  // get the embeddings
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  // query back-end database using the embeddings
  const results = await index.query({
    topK: 5,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  // create a string to send the results to openAI
  let resultString = "\n\nReturned Results from Pinecone Vector Database: ";
  results.matches.forEach((match) => {
    resultString += `\n
  Professor: ${match.id}
  Review: ${match.metadata.stars}
  Subject: ${match.metadata.subject}
  Stars: ${match.metadata.stars}
  \n\n`;
  });

  // combine the user’s question with the Pinecone results
  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  // send chat completion request to OpenAI
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  // handle the streaming response using a readable stream
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
