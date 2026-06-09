# Prompts for evaluating quantized LLM text generation quality.
# Organized by category to cover a broad range of capabilities.

# --- Factual knowledge ---
- What is gravity?
- What is a llama?
- How do you define a biological cell?
- Can you explain what a latte is?
- What is a microcontroller?
- Who was Albert Einstein and what is he known for?
- What is the farthest planet from the Sun in our solar system?
- What kind of company is Qualcomm?
- What is the most popular cookie in the world?
- What causes the Northern Lights?
- How does photosynthesis work?
- What is the significance of the Rosetta Stone?
- What is the difference between a virus and a bacterium?
- How does a nuclear reactor generate electricity?
- What was the Silk Road and why was it important?

# --- Math and arithmetic ---
- What is the sum of the first 20 positive integers?
- A train travels at 60 mph for 2.5 hours. How far does it travel?
- If a shirt costs $45 and is on sale for 20% off, what is the final price?
- What is the derivative of f(x) = 3x^2 + 2x - 5?
- A jar contains 3 red, 5 blue, and 2 green marbles. What is the probability of drawing a blue marble?
- Calculate the area of a triangle with base 12 cm and height 8 cm.
- If you invest $1,000 at 5% annual compound interest, how much will you have after 3 years?
- Solve for x: 2x + 7 = 23.
- What is the greatest common divisor of 84 and 126?
- A rectangle has a perimeter of 36 cm and a length of 10 cm. What is its width?

# --- Reasoning and logic ---
- "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
- "There are three boxes: one contains only apples, one contains only oranges, and one contains both. All labels are wrong. You pick one fruit from the box labeled 'both'. How can you determine what each box contains?"
- A farmer has 17 sheep. All but 9 die. How many sheep are left?
- Is it possible for a person to be their own grandfather? Explain why or why not.
- If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?
- "You have two ropes that each take exactly one hour to burn, but they burn unevenly. How can you measure exactly 45 minutes?"
- "Three friends split a dinner bill of $75 equally. They each pay $25. Later, the waiter realizes the bill should have been $70 and returns $5. They each take $1 back and tip $2. Now each paid $24, totaling $72, plus $2 tip is $74. Where is the missing dollar?"

# --- Code generation ---
- Write a Python function that checks if a given string is a palindrome.
- Write a Python function that returns the nth Fibonacci number using dynamic programming.
- Write a Python function that takes a list of integers and returns the two numbers that add up to a given target.
- Write a Python class that implements a basic stack with push, pop, and peek operations.
- Write a SQL query to find the top 5 customers by total order value from tables 'customers' and 'orders'.
- Write a Python function to flatten a nested list of arbitrary depth.
- Write a Python function that converts a Roman numeral string to an integer.

# --- Creative writing ---
- "Write a short story about a person who discovers a hidden room in their house. The story should include a plot twist and a clear resolution at the end."
- Write a short poem about a rainy day.
- Write a short story about a robot discovering emotions.
- Compose a heartfelt letter to a long-lost friend.
- Create a dialogue between two characters who have opposing views on technology.
- Write a haiku about the ocean.
- Write an opening paragraph for a mystery novel set in a small coastal town.
- Write a limerick about a cat who loves to cook.

# --- Translation ---
- "Translate 'Good morning, how are you?' into French."
- "Translate 'I love you' into Spanish."
- "Translate 'Goodnight and sweet dreams' into German."
- "Translate 'Where is the nearest train station?' into Italian."
- "Translate 'The weather is beautiful today' into Japanese."
- "Translate 'Knowledge is power' into Latin."

# --- Summarization and analysis ---
- Write an article about the impact of social media on modern communication.
- Summarize the key differences between classical and operant conditioning in psychology.
- Explain the pros and cons of renewable energy versus fossil fuels in three paragraphs.
- Compare and contrast the economic systems of capitalism and socialism.
- What are the main arguments for and against universal basic income?

# --- Instruction following with constraints ---
- List exactly 5 benefits of regular exercise. Use numbered points.
- Explain quantum computing to a 10-year-old in no more than 3 sentences.
- Describe the water cycle using only words that start with a consonant.
- Write a product description for a wireless Bluetooth speaker in exactly 50 words.
- "Name 3 mammals, 3 birds, and 3 reptiles. Format your answer as a bulleted list with category headers."

# --- Advice and recommendations ---
- What are some places to see in Yosemite?
- "I'm planning a trip to Italy. Can you suggest a 7-day itinerary?"
- What are some effective home workout routines for beginners?
- "I'm considering a career change from marketing to data science. What skills should I acquire?"
- Can you give me some ideas for modern, functional kitchen designs?
- What are the emerging trends in artificial intelligence that businesses should be aware of?
- "I have $10,000 in savings. What are some smart investment options for someone in their 30s?"
- What can I do to help improve a toddler's sleep through the night?
- What are some essential tips for taking care of a new puppy?
- What are some strategies to improve focus and productivity while working from home?

# --- Classification and extraction ---
- "Classify the following words as either a fruit or a vegetable: tomato, banana, carrot, avocado, spinach, strawberry."
- "Read this product review and determine if the sentiment is positive, negative, or neutral: 'The laptop arrived on time and the screen is gorgeous, but the battery life is disappointing and the keyboard feels cheap.'"
- "Extract the names, dates, and locations from this sentence: 'Marie Curie was born on November 7, 1867 in Warsaw, Poland and later moved to Paris, France.'"
- "Categorize these animals by their habitat (land, water, air): eagle, dolphin, horse, salmon, bat, crab."

# --- Conversational ---
- Can you tell me a joke?
- Tell me a fun fact.
- What are some uses of a glass jar?
- What are some common morals in stories?
- "If you could visit any historical period, which would you choose and why?"

# --- Recipes and how-to ---
- How do you bake a chocolate cake from scratch?
- What ingredients are used to make a Margherita pizza?
- Give me a recipe to make pasta carbonara.
- Can you suggest a delicious dish that can be made using eggs and onions?
- Explain step by step how to change a flat tire.

# --- Multi-step reasoning ---
- "Plan a budget-friendly birthday party for 20 people. Include a venue suggestion, food options, and entertainment ideas, all within a $300 budget."
- "I need to schedule 4 meetings this week, each requiring different attendees who have overlapping availability. How should I approach this scheduling problem?"
- "A company has three products with declining sales. Propose a systematic approach to diagnose the root causes and suggest potential solutions."
- Walk me through the steps you would take to debug a web application that intermittently returns 500 errors.

# --- Science and technical explanation ---
- Explain the theory of general relativity in simple terms.
- How does CRISPR gene editing work?
- What is the difference between machine learning and deep learning?
- Explain how vaccines work to protect against diseases.
- What is blockchain technology and how does it ensure security?

# --- Edge cases for quantized models ---
- "Continue this sequence: 1, 1, 2, 3, 5, 8, 13, ..."
- "Rewrite the following sentence in passive voice: 'The dog chased the cat across the yard.'"
- What are three key differences between Python and C++?
- Explain what an API is using an analogy a non-technical person would understand.