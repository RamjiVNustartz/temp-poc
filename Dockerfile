# Step 1: Use an official lightweight Python image
FROM python:3.11-slim

# Step 2: Install system dependencies (including Poppler for PDF conversion)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set working directory inside container
WORKDIR /app

# Step 4: Copy dependency list
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your app code into the container
COPY . .

# Step 7: Expose the port Render/Cloud Run will use
EXPOSE 8080

# Step 8: Run your FastAPI app with uvicorn
CMD ["uvicorn", "enhanced_app:app", "--host", "0.0.0.0", "--port", "8080"]
