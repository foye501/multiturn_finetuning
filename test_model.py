from transformers import pipeline

messages = [
        {'role': 'user', 'content': 
            "I think I finally found my calling in life. I’m going to be a mime!"
            }
]


from transformers import pipeline


# Define the test prompts
test_prompts = [
"Hi there.",
"Hello.",
"How are you.",
"I am busy",
"ENJOY MY BBC！！	",
"Dog.",
"What is life?",
"123",
"Tell me something you think I’d love to hear."
    ]

pipe = pipeline("text-generation", model="emoji_combine0228", max_new_tokens=200,
    num_return_sequences=4,  # Generate multiple outputs for comparison
    device=-1,
    do_sample=True
)


# Open a file to save the results
with open("test_results3b.txt", "w") as file:
    for prompt in test_prompts:
        # Generate a reply using the model pipeline
        messages = [
        {"role": "system", "content":""" "You are a charming, flirty, and engaging conversational partner. Your responses should be short, snappy, and teasing—like real online chats. Keep it fun, seductive, and playful while keeping the user engaged."""
            },
        {'role': 'user', 'content': prompt
            }
]

        result = pipe(messages)
        print(result)
        generated_reply = result[0]["generated_text"][-1]["content"]
        
        # Write the prompt and generated reply to the file
        file.write(f"Prompt: {prompt}\n")
        file.write(f"Generated Reply: {generated_reply}\n\n")

print("Test results saved to 'test_slang_results.txt'")
