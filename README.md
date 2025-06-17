# chatPDF


## How to Run

To run the chatPDF. App locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/skilobyte/chatPDF.git
   ```
2. **Navigate to the Directory:**
   ```bash
   cd chatPDF
   ```
3. **Install Python Dependencies:**
    ```bash
   pip install -r requirements.txt
   ```
4. **Configure OpenAI API Key:**

Obtain an API key from OpenAI [here](https://platform.openai.com/account/api-keys).

Set the key as an environment variable before running the backend:

    ```bash
    export OPENAI_API_KEY="sk-..."  # Replace with your actual key
    ```
5. **Run Backend:**
    ```bash
   python main.py
   ```
    This will start the Backend Flask server on port  ```5000```
6. **Run Frontend:**
    ```bash
        python3 -m http.server 5500     
    ```   