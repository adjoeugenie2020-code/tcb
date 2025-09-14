import axios from 'axios';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export async function askHuggingFace(question: string): Promise<string> {
  try {
    if (!question || typeof question !== "string") {
      return "Message invalide. Veuillez réessayer.";
    }

    const response = await axios.post(
      'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2',
      {
        inputs: question,
        parameters: {
          max_new_tokens: 200,
          temperature: 0.7,
          do_sample: true,
          return_full_text: false
        }
      },
      {
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_HF_API_KEY}`,
          'Content-Type': 'application/json',
        }
      }
    );

    if (response.data && Array.isArray(response.data) && response.data.length > 0) {
      return response.data[0].generated_text || 'Désolé, je n\'ai pas pu générer une réponse.';
    } else if (response.data && response.data.generated_text) {
      return response.data.generated_text;
    } else {
      return 'Désolé, je n\'ai pas pu comprendre votre question.';
    }
  } catch (error) {
    console.error('Erreur lors de l\'appel à Hugging Face:', error);
    if (axios.isAxiosError(error) && error.response?.status === 503) {
      return 'Le modèle est en cours de chargement, veuillez réessayer dans quelques secondes.';
    }
    return 'Désolé, une erreur s\'est produite. Veuillez réessayer plus tard.';
  }
}

// Fonction de fallback Groq (gardée pour compatibilité)
export async function askGroq(message: string): Promise<string> {
  try {
    if (!message || typeof message !== "string") {
      return "Message invalide. Veuillez réessayer.";
    }

    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${import.meta.env.VITE_GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "llama3-70b-8192",
        messages: [
          { role: "system", content: "Tu es un assistant utile qui répond en français de manière claire et concise." },
          { role: "user", content: message }
        ],
        max_tokens: 1024,
        temperature: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.choices && data.choices.length > 0) {
      return data.choices[0].message.content;
    } else {
      throw new Error('Aucune réponse reçue du modèle');
    }
  } catch (error) {
    console.error('Erreur lors de l\'appel à Groq:', error);
    return 'Désolé, une erreur s\'est produite. Veuillez réessayer plus tard.';
  }
}