# Step 1: Use an official lightweight Python image
FROM python:3.11-slim

# Step 2: Set working directory inside container
WORKDIR /app

# Step 3: Copy dependency list
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your app code into the container
COPY . .

# Step 6: Expose the port Cloud Run will use
EXPOSE 8080

# Step 7: Run your FastAPI app with uvicorn
CMD ["uvicorn", "enhanced_app:app", "--host", "0.0.0.0", "--port", "8080"]
