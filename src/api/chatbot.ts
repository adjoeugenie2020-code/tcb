export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

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

// Fonction de fallback Hugging Face (optionnelle)
export async function askHuggingFace(question: string): Promise<string> {
  try {
    const response = await fetch(
      'https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_HF_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          inputs: question,
          parameters: {
            max_length: 100,
            temperature: 0.7,
            do_sample: true,
          },
        }),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    if (Array.isArray(data) && data.length > 0) {
      return data[0].generated_text || data[0].text || 'Désolé, je n\'ai pas pu générer une réponse.';
    } else if (data.generated_text) {
      return data.generated_text;
    } else if (data.text) {
      return data.text;
    } else {
      return 'Désolé, je n\'ai pas pu comprendre votre question.';
    }
  } catch (error) {
    console.error('Erreur lors de l\'appel à Hugging Face:', error);
    return 'Désolé, une erreur s\'est produite. Veuillez réessayer plus tard.';
  }
}
