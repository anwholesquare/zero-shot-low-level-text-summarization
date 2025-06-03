# XLSUM Text Summarization API

A specialized Flask-based API for processing local XLSUM dataset files and generating text summaries using multiple AI services. This API focuses on low-level text summarization research with support for Bengali, Nepali, Burmese, and Sinhala languages.

## Features

- **Local XLSUM Dataset Loading**: Direct loading from local JSONL files (no internet required)
- **Multi-Language Support**: Bengali, Nepali, Burmese, and Sinhala
- **Multiple AI Services**: OpenAI, Anthropic Claude, DeepSeek, and Bloomz
- **Three Prompt Strategies**: Direct, minimal details, and analytical details
- **Comprehensive Evaluation**: ROUGE-1, BLEU, and BERTScore metrics
- **Queue-Based Processing**: Background processing with resume functionality
- **CSV Sample Export**: Save sampled data for later processing
- **Progress Monitoring**: Real-time progress tracking with time estimation
- **File-Based Persistence**: No database required, all data stored in files
- **Resume Capability**: Resume interrupted processing from any batch

## New Queue-Based Workflow

The API now supports a queue-based workflow for efficient processing:

1. **Save Samples**: Extract and save random samples to CSV files
2. **Start Summarization**: Queue background summarization jobs
3. **Monitor Progress**: Track progress with real-time updates
4. **Resume Processing**: Continue from where you left off if interrupted

## Dataset Requirements

The API expects local XLSUM dataset files in JSONL format at these locations:

```
XLSUM/bengali_train.jsonl
XLSUM/burmese_train.jsonl
XLSUM/nepali_train.jsonl
XLSUM/sinhala_train.jsonl
```

Each JSONL file should contain one JSON object per line with the following structure:
```json
{
  "id": "unique_identifier",
  "url": "source_url",
  "title": "article_title",
  "summary": "human_written_summary",
  "text": "full_article_content"
}
```

**Required fields**: `text`, `summary`  
**Optional fields**: `id`, `url`, `title`

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- API keys for AI services
- Local XLSUM dataset files (see Dataset Requirements above)
- (Optional) CUDA for GPU acceleration with Bloomz

### Setup

1. **Clone and setup environment**
```bash
git clone <repository-url>
cd xlsum-text-summarization
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Test local data loading**
```bash
python test_local_data.py
```
This will verify that your XLSUM files are properly formatted and accessible.

4. **Download NLTK data (for BLEU scoring)**
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **Download BERT models (for BERTScore - will download automatically on first use)**
```bash
# BERTScore will automatically download required BERT models on first use
# This may take a few minutes for the initial setup
```

6. **Configure environment variables**
```bash
# Create .env file with your API keys (this file is git-ignored for security)
touch .env

# Add your API keys to .env file:
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" >> .env
echo "DEEPSEEK_API_KEY=your_deepseek_api_key_here" >> .env
echo "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1" >> .env
echo "GEMINI_API_KEY=your_gemini_api_key_here" >> .env

# Or edit .env manually with your preferred editor:
# nano .env
```

**Important**: The `.env` file is automatically ignored by git (see `.gitignore`) to protect your API keys from being committed to the repository.

7. **Test AI services**
```bash
python test_services.py
```

8. **Run the application**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /
```
Returns API status, supported languages, and available services.

### Queue-Based Endpoints (New)

#### Save Sample Data
```
POST /api/save_sample
```
Extract and save random samples to CSV files for later processing.

**Request Body:**
```json
{
  "lang": ["bengali", "nepali"]  // or single string "bengali"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Sampled data saved for 2 languages",
  "languages": ["bengali", "nepali"],
  "saved_files": {
    "bengali": {
      "filename": "samples_bengali_20240115_103000.csv",
      "filepath": "data/samples/samples_bengali_20240115_103000.csv",
      "sample_count": 2000,
      "columns": ["index", "language", "id", "title", "text", "summary", "url", "text_length", "summary_length", "created_at"]
    }
  },
  "total_samples": 4000,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Start Summarization Queue
```
POST /api/summarize
```
Start background summarization process with queue management.

**Request Body:**
```json
{
  "lang": "bengali",                    // Required: language
  "service": "openai",                  // Required: AI service
  "batch_size": 10,                     // Optional: samples per batch (default: 10)
  "max_token": 7000,                    // Optional: max tokens (default: 7000)
  "resume_batch": 0,                    // Optional: batch to resume from (default: 0)
  "prompt_type": "direct",              // Optional: prompt template (default: "direct")
  "samples_file": "samples_bengali_20240115_103000.csv"  // Optional: specific file
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Summarization started",
  "queue_id": "a1b2c3d4",
  "language": "bengali",
  "service": "openai",
  "batch_size": 10,
  "max_tokens": 7000,
  "resume_batch": 0,
  "prompt_type": "direct",
  "samples_file": "data/samples/samples_bengali_20240115_103000.csv",
  "created_at": "2024-01-15T10:30:00"
}
```

#### Show Queue Progress
```
GET /api/show_progress/<queue_id>
```
Monitor the progress of a summarization queue.

**Response:**
```json
{
  "queue_id": "a1b2c3d4",
  "queue_data": {
    "status": "processing",
    "language": "bengali",
    "service": "openai",
    "progress_percentage": 45.5,
    "processed_samples": 910,
    "total_samples": 2000,
    "current_batch": 91,
    "total_batches": 200,
    "elapsed_time_seconds": 1250.5,
    "estimated_remaining_seconds": 1500.2,
    "estimated_completion": "2024-01-15T11:15:00",
    "last_batch_completed": 90,
    "created_at": "2024-01-15T10:30:00",
    "started_at": "2024-01-15T10:30:15"
  },
  "timestamp": "2024-01-15T10:50:30"
}
```

#### Stop Queue
```
POST /api/stop_queue/<queue_id>
```
Stop/cancel a running or queued summarization process.

**Response (Immediate cancellation):**
```json
{
  "status": "success",
  "message": "Queue a1b2c3d4 cancelled successfully",
  "queue_id": "a1b2c3d4",
  "previous_status": "queued",
  "new_status": "cancelled",
  "cancelled_at": "2024-01-15T10:45:00"
}
```

**Response (Graceful cancellation during processing):**
```json
{
  "status": "success",
  "message": "Queue a1b2c3d4 marked for cancellation. It will stop after the current batch.",
  "queue_id": "a1b2c3d4",
  "previous_status": "processing",
  "note": "The queue will stop gracefully after completing the current batch",
  "cancellation_requested_at": "2024-01-15T10:45:00"
}
```

**Error Response:**
```json
{
  "error": "Cannot stop queue a1b2c3d4. Current status: completed",
  "current_status": "completed"
}
```

#### List All Queues
```
GET /api/list_queues
```
List all queues (active and completed).

**Response:**
```json
{
  "status": "success",
  "total_queues": 5,
  "queues": [
    {
      "queue_id": "a1b2c3d4",
      "status": "processing",
      "language": "bengali",
      "service": "openai",
      "created_at": "2024-01-15T10:30:00",
      "progress_percentage": 45.5,
      "source": "active"
    }
  ],
  "timestamp": "2024-01-15T10:50:30"
}
```

### Legacy Endpoints (Still Available)

#### Load Datasets
```
POST /api/load-datasets
```
Loads and samples XLSUM datasets from local JSONL files.

**Response:**
```json
{
  "status": "success",
  "message": "Datasets loaded and sampled successfully",
  "datasets": {
    "bengali": 2000,
    "nepali": 2000,
    "burmese": 2000,
    "sinhala": 2000
  }
}
```

**Note**: This endpoint loads data from `XLSUM/{language}_train.jsonl` files and randomly samples 2000 items from each dataset.

### Generate Summaries (Batch)
```
POST /api/generate-summaries
```
Generate summaries for a specific batch of samples.

**Request Body:**
```json
{
  "language": "bengali",
  "prompt_type": "direct",
  "service": "openai",
  "batch_size": 10,
  "start_index": 0
}
```

**Response:**
```json
{
  "status": "success",
  "processed_count": 10,
  "total_items": 2000,
  "start_index": 0,
  "end_index": 10,
  "results": [
    {
      "id": "bengali_0_openai_direct",
      "generated_summary": "AI-generated summary text...",
      "usual_summary": "Original human summary...",
      "scoring_rouge": {
        "rouge1": 0.45
      },
      "scoring_bleu": 0.32,
      "scoring_bertscore": 0.85,
      "source_url": "https://www.bbc.com/bengali/...",
      "title": "Article title",
      "content": "Full article text...",
      "language": "bengali",
      "prompt_type": "direct",
      "service": "openai",
      "timestamp": "2024-01-15T10:30:00"
    }
  ]
}
```

### Batch Process Entire Dataset
```
POST /api/batch-process
```
Process an entire language dataset with specified parameters.

**Request Body:**
```json
{
  "language": "sinhala",
  "prompt_type": "analytical_details",
  "services": ["openai", "anthropic"]
}
```

### Export to CSV
```
POST /api/export-csv
```
Convert JSON results to CSV format.

**Request Body:**
```json
{
  "results_file": "results_bengali_direct_20240115_103000.json"
}
```

### Dataset Information
```
GET /api/dataset-info
```
Get information about loaded datasets and service status.

## Prompt Templates

### 1. Direct
```
Summarize the following text:

{text}

Summary:
```

### 2. Minimal Details
```
Please provide a concise summary of the following text. Focus on the main points and key information.

Text: {text}

Summary:
```

### 3. Analytical Details
```
Analyze the following text and provide a comprehensive summary that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:
```

## Usage Examples

### Queue-Based Workflow (Recommended)

```python
import requests
import time

BASE_URL = "http://localhost:5000"

# Step 1: Save sample data to CSV
print("Saving sample data...")
response = requests.post(f"{BASE_URL}/api/save_sample", json={
    "lang": ["bengali", "nepali"]
})
result = response.json()
print(f"Saved {result['total_samples']} samples")

# Step 2: Start summarization queue
print("Starting summarization...")
response = requests.post(f"{BASE_URL}/api/summarize", json={
    "lang": "bengali",
    "service": "openai",
    "batch_size": 20,
    "max_token": 150,
    "prompt_type": "minimal_details"
})
queue_result = response.json()
queue_id = queue_result['queue_id']
print(f"Started queue: {queue_id}")

# Step 3: Monitor progress
print("Monitoring progress...")
while True:
    response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
    progress = response.json()['queue_data']
    
    status = progress['status']
    percentage = progress.get('progress_percentage', 0)
    processed = progress.get('processed_samples', 0)
    total = progress.get('total_samples', 0)
    
    print(f"Status: {status}, Progress: {percentage:.1f}%, Samples: {processed}/{total}")
    
    if status in ['completed', 'error']:
        break
    
    time.sleep(10)  # Check every 10 seconds

print("Processing completed!")

# Step 4: Resume if needed (example)
if status == 'error':
    print("Resuming from last completed batch...")
    last_batch = progress.get('last_batch_completed', 0)
    
    response = requests.post(f"{BASE_URL}/api/summarize", json={
        "lang": "bengali",
        "service": "openai",
        "batch_size": 20,
        "resume_batch": last_batch + 1,  # Resume from next batch
        "samples_file": "samples_bengali_20240115_103000.csv"
    })
    new_queue_id = response.json()['queue_id']
    print(f"Resumed with new queue: {new_queue_id}")
```

### Queue Stopping Example

```python
import requests
import time

BASE_URL = "http://localhost:5000"

# Start a queue
response = requests.post(f"{BASE_URL}/api/summarize", json={
    "lang": "bengali",
    "service": "openai",
    "batch_size": 10,
    "prompt_type": "direct"
})

queue_id = response.json()['queue_id']
print(f"Started queue: {queue_id}")

# Monitor for a while
for i in range(5):
    response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
    progress = response.json()['queue_data']
    
    print(f"Status: {progress['status']}, Progress: {progress.get('progress_percentage', 0):.1f}%")
    
    if progress['status'] == 'processing':
        break
    time.sleep(2)

# Stop the queue
print(f"Stopping queue {queue_id}...")
response = requests.post(f"{BASE_URL}/api/stop_queue/{queue_id}")

if response.status_code == 200:
    result = response.json()
    print(f"✅ {result['message']}")
    
    # Check final status
    time.sleep(2)
    response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
    final_status = response.json()['queue_data']['status']
    print(f"Final status: {final_status}")
else:
    print(f"❌ Failed to stop: {response.json()}")
```

### Batch Processing Multiple Languages

```python
# Process multiple languages with different services
languages = ["bengali", "nepali", "burmese", "sinhala"]
services = ["openai", "anthropic", "deepseek"]

# Save samples for all languages
response = requests.post(f"{BASE_URL}/api/save_sample", json={
    "lang": languages
})

# Start queues for each combination
queue_ids = []
for lang in languages:
    for service in services:
        response = requests.post(f"{BASE_URL}/api/summarize", json={
            "lang": lang,
            "service": service,
            "batch_size": 15,
            "prompt_type": "analytical_details"
        })
        if response.status_code == 200:
            queue_id = response.json()['queue_id']
            queue_ids.append((queue_id, lang, service))
            print(f"Started {lang}-{service}: {queue_id}")

# Monitor all queues
print(f"Monitoring {len(queue_ids)} queues...")
while queue_ids:
    completed = []
    for queue_id, lang, service in queue_ids:
        response = requests.get(f"{BASE_URL}/api/show_progress/{queue_id}")
        if response.status_code == 200:
            progress = response.json()['queue_data']
            status = progress['status']
            percentage = progress.get('progress_percentage', 0)
            
            print(f"{lang}-{service} ({queue_id}): {status} - {percentage:.1f}%")
            
            if status in ['completed', 'error']:
                completed.append((queue_id, lang, service))
    
    # Remove completed queues
    for item in completed:
        queue_ids.remove(item)
        print(f"Completed: {item[1]}-{item[2]}")
    
    if queue_ids:
        time.sleep(30)  # Check every 30 seconds

print("All processing completed!")
```

### Legacy Workflow (Still Available)

### Python Client Example
```python
import requests

# Load datasets
response = requests.post('http://localhost:5000/api/load-datasets')
print(response.json())

# Generate summaries for Bengali texts using OpenAI
response = requests.post('http://localhost:5000/api/generate-summaries', json={
    'language': 'bengali',
    'prompt_type': 'direct',
    'service': 'openai',
    'batch_size': 5,
    'start_index': 0
})
results = response.json()

# Process results
for result in results['results']:
    print(f"Title: {result['title']}")
    print(f"Generated: {result['generated_summary'][:100]}...")
    print(f"ROUGE-1: {result['scoring_rouge']['rouge1']:.3f}")
    print("-" * 50)
```

### Batch Processing Example
```python
# Process entire Nepali dataset with multiple services
response = requests.post('http://localhost:5000/api/batch-process', json={
    'language': 'nepali',
    'prompt_type': 'minimal_details',
    'services': ['openai', 'anthropic', 'deepseek']
})

batch_result = response.json()
print(f"Processed {batch_result['total_processed']} summaries")
print(f"Output file: {batch_result['output_file']}")

# Export to CSV
csv_response = requests.post('http://localhost:5000/api/export-csv', json={
    'results_file': batch_result['output_file']
})
print(f"CSV exported: {csv_response.json()['csv_file']}")
```

## Evaluation Metrics

The API automatically calculates the following metrics for each generated summary with **language-specific support**:

- **ROUGE-1**: Unigram overlap between generated and reference summaries (language-agnostic with proper tokenization)
- **BLEU**: Bilingual evaluation understudy score (language-agnostic)
- **BERTScore**: Contextual embeddings-based evaluation using language-specific BERT models

### Language-Specific BERTScore

The system uses appropriate BERT models for each supported language:

| Language | BERTScore Code | Model Used |
|----------|----------------|------------|
| Bengali  | `bn` | Bengali BERT model |
| Nepali   | `ne` | Nepali BERT model |
| Burmese  | `my` | Burmese BERT model |
| Sinhala  | `si` | Sinhala BERT model |
| English  | `en` | English BERT model (fallback) |

**Fallback Mechanism**: If a language-specific BERT model is not available or fails to load, the system automatically falls back to the English BERT model to ensure metrics are always calculated.

### Metrics Calculation Features

- **Automatic Language Detection**: Metrics are calculated using the appropriate language model based on the dataset language
- **Robust Error Handling**: Graceful fallback to English models if language-specific models fail
- **Consistent Results**: Deterministic calculation ensuring reproducible results
- **Performance Logging**: Detailed logging of which models are used for transparency

## Output Format

Generated datasets follow this structure:

```json
{
  "id": "language_index_service_prompttype",
  "generated_summary": "AI-generated summary with max 7000 tokens",
  "usual_summary": "Original human-written summary",
  "scoring_rouge": {
    "rouge1": 0.45
  },
  "scoring_bleu": 0.32,
  "scoring_bertscore": 0.85,
  "source_url": "https://www.bbc.com/...",
  "title": "Article title",
  "content": "Full article content",
  "language": "bengali|nepali|burmese|sinhala",
  "prompt_type": "direct|minimal_details|analytical_details",
  "service": "openai|anthropic|deepseek|bloomz",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_xlsum_api.py
```

This will test:
- Dataset loading and sampling
- Summary generation with different configurations
- Metric calculation
- Service availability

### Test Language-Specific Metrics

Test the language-specific metrics calculation:

```bash
python test_language_metrics.py
```

This will test:
- BERTScore calculation for Bengali, Nepali, Burmese, and Sinhala
- ROUGE and BLEU scores for all languages
- English fallback mechanism for unsupported languages
- Metrics consistency across multiple runs

### Test Queue Stopping

Test the queue stopping functionality:

```bash
python test_stop_queue.py
```

This will test:
- Starting and stopping queues
- Graceful cancellation during processing
- Queue state management

## Performance Considerations

- **Dataset Loading**: Initial loading takes 2-5 minutes depending on internet speed
- **Bloomz Model**: Requires 2-4GB RAM, GPU recommended for faster inference
- **API Rate Limits**: Respect rate limits for external services
- **Batch Processing**: Large batches may take hours to complete
- **Storage**: Results files can be large (100MB+ for full datasets)

## Research Applications

This API is designed for:

- **Cross-lingual summarization research**
- **Low-resource language processing**
- **Prompt engineering studies**
- **AI model comparison and evaluation**
- **Multilingual NLP benchmarking**

## File Structure

```
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file (protects .env and data files)
├── test_xlsum_api.py        # Legacy test suite
├── test_queue_api.py        # Queue-based test suite (New)
├── test_local_data.py       # Local data loading test
├── setup_directories.py    # Directory setup script
├── README.md                # This documentation
├── services/                # AI service implementations
│   ├── openai_service.py
│   ├── anthropic_service.py
│   ├── deepseek_service.py
│   └── bloomz_service.py
├── data/                    # Data storage directory
│   ├── samples/             # CSV sample files (New)
│   │   ├── samples_bengali_20240115_103000.csv
│   │   ├── samples_nepali_20240115_103000.csv
│   │   └── ...
│   ├── queues/              # Queue state files (New)
│   │   ├── a1b2c3d4.json   # Queue state
│   │   ├── e5f6g7h8.json
│   │   └── ...
│   ├── results/             # Processing results (New)
│   │   ├── results_a1b2c3d4.json
│   │   ├── results_e5f6g7h8.json
│   │   └── ...
│   └── [legacy files]       # Legacy result files
├── logs/                    # Application logs
│   └── xlsum_processing.log
├── XLSUM/                   # Local dataset files
│   ├── bengali_train.jsonl
│   ├── nepali_train.jsonl
│   ├── burmese_train.jsonl
│   └── sinhala_train.jsonl
└── .env                     # Environment variables (git-ignored)
```

## Queue Management

### Queue States
- **queued**: Queue created, waiting to start
- **started**: Worker thread started
- **processing**: Actively processing samples
- **completed**: All samples processed successfully
- **cancelled**: Queue stopped by user request
- **error**: Processing failed with error

### Resume Functionality
The API supports resuming interrupted processing:

1. **Automatic Resume**: If a queue fails, you can resume from the last completed batch
2. **Manual Resume**: Specify `resume_batch` parameter to start from a specific batch
3. **State Persistence**: Queue state is saved to files for recovery after restarts
4. **Progress Tracking**: Detailed progress information including time estimates

### File-Based Storage
- **No Database Required**: All data stored in JSON and CSV files
- **Queue States**: Stored in `data/queues/{queue_id}.json`
- **Sample Data**: Stored in `data/samples/samples_{lang}_{timestamp}.csv`
- **Results**: Stored in `data/results/results_{queue_id}.json`
- **Logs**: Detailed processing logs in `logs/xlsum_processing.log`

## API Key Setup

### OpenAI
1. Visit [OpenAI API](https://platform.openai.com/api-keys)
2. Create an account and generate an API key
3. Add `OPENAI_API_KEY=your_key` to `.env`

### Anthropic
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account and generate an API key
3. Add `ANTHROPIC_API_KEY=your_key` to `.env`

### DeepSeek
1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Create an account and generate an API key
3. Add `DEEPSEEK_API_KEY=your_key` to `.env`

### Google Gemini
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an account and generate an API key
3. Add `GEMINI_API_KEY=your_key` to `.env`

### Bloomz (Local)
No API key required - uses local Hugging Face transformers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this API in your research, please cite:

```bibtex
@misc{xlsum-api-2024,
  title={XLSUM Text Summarization API},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/xlsum-text-summarization}
}
```

Also cite the original XLSUM dataset:

```bibtex
@inproceedings{hasan-etal-2021-xl,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Islam, Md. Saiful  and
      Mubasshir, Kazi  and
      Li, Yuan-Fang  and
      Kang, Yong-Bin  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.413",
    pages = "4693--4703",
}
``` 