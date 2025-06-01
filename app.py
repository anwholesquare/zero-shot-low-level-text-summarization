from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import random
from loguru import logger
from typing import Dict, List, Any
import json
from datetime import datetime
import threading
import uuid
import time
from pathlib import Path
from dotenv import load_dotenv
import zipfile
import tempfile
import shutil

# Load environment variables from .env file
load_dotenv()

# Import AI service clients
from services.openai_service import OpenAIService
from services.anthropic_service import AnthropicService
from services.deepseek_service import DeepSeekService
from services.bloomz_service import BloomzService

# Import evaluation metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import evaluate
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("Evaluation metrics not available. Install rouge-score, nltk, evaluate, and bert-score packages.")
    METRICS_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logger.add("logs/xlsum_processing.log", rotation="1 day", retention="7 days", level="INFO")

# Initialize AI services
openai_service = OpenAIService()
anthropic_service = AnthropicService()
deepseek_service = DeepSeekService()
bloomz_service = BloomzService()

# Global variables for dataset storage
xlsum_datasets = {}
processed_datasets = {}

# Queue management
active_queues = {}
queue_lock = threading.Lock()
cancelled_queues = set()  # Track cancelled queue IDs

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('data/samples', exist_ok=True)
os.makedirs('data/queues', exist_ok=True)
os.makedirs('data/results', exist_ok=True)

# Prompt templates for summarization - Language-specific
PROMPT_TEMPLATES = {
    "bengali": {
        "direct": "Summarize the following text in Bangla within 350 letters:\n\n{text}\n\nSummary:",
        
        "minimal_details": """Please provide a concise summary of the following text within 350 letters in Bangla. Focus on the main points and key information.

Text: {text}

Summary:""",
        
        "analytical_details": """Analyze the following text and provide a summary within 350 letters in Bangla that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:"""
    },
    
    "nepali": {
        "direct": "Summarize the following text in Nepali within 350 letters:\n\n{text}\n\nSummary:",
        
        "minimal_details": """Please provide a concise summary of the following text within 350 letters in Nepali. Focus on the main points and key information.

Text: {text}

Summary:""",
        
       "analytical_details": """Analyze the following text and provide a summary within 350 letters in Nepali that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:"""
    },
    
    "burmese": {
        "direct": "Summarize the following text in Burmese within 350 letters:\n\n{text}\n\nSummary:",
        
        "minimal_details": """Please provide a concise summary of the following text within 350 letters in Burmese. Focus on the main points and key information.

Text: {text}

Summary:""",
        
       "analytical_details": """Analyze the following text and provide a summary within 350 letters in Burmese that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:"""
    },
    
    "sinhala": {
        "direct": "Summarize the following text in Sinhala within 350 letters:\n\n{text}\n\nSummary:",
        
        "minimal_details": """Please provide a concise summary of the following text within 350 letters in Sinhala. Focus on the main points and key information.

Text: {text}

Summary:""",
        
        "analytical_details": """Analyze the following text and provide a summary within 350 letters in Sinhala that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:"""
    },
    
    # English fallback for any other languages
    "english": {
        "direct": "Summarize the following text within 350 letters:\n\n{text}\n\nSummary:",
        
        "minimal_details": """Please provide a concise summary of the following text within 350 letters. Focus on the main points and key information.

Text: {text}

Summary:""",
        
        "analytical_details": """Analyze the following text and provide a summary within 350 letters that captures:
1. Main topic and key points
2. Important details and context
3. Conclusions or outcomes mentioned

Text: {text}

Detailed Summary:"""
    }
}

def get_prompt_template(language: str, prompt_type: str) -> str:
    """Get language-specific prompt template"""
    # Map language names to template keys
    lang_map = {
        "bengali": "bengali",
        "nepali": "nepali", 
        "burmese": "burmese",
        "sinhala": "sinhala"
    }
    
    # Get the appropriate language key, fallback to English
    lang_key = lang_map.get(language.lower(), "english")
    
    # Get the prompt template, fallback to direct if prompt_type not found
    if lang_key in PROMPT_TEMPLATES and prompt_type in PROMPT_TEMPLATES[lang_key]:
        return PROMPT_TEMPLATES[lang_key][prompt_type]
    else:
        # Fallback to English
        return PROMPT_TEMPLATES["english"].get(prompt_type, PROMPT_TEMPLATES["english"]["direct"])

def load_xlsum_languages():
    """Load XLSUM datasets from local JSONL files"""
    languages = ['bengali', 'nepali', 'burmese', 'sinhala']
    
    logger.info("Loading XLSUM datasets from local files...")
    
    for lang in languages:
        try:
            logger.info(f"Loading {lang} dataset...")
            file_path = f"XLSUM/{lang}_train.jsonl"
            
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                xlsum_datasets[lang] = None
                continue
            
            # Load JSONL file
            dataset_items = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        dataset_items.append(item)
            
            xlsum_datasets[lang] = dataset_items
            logger.info(f"Loaded {len(dataset_items)} samples for {lang}")
            
        except Exception as e:
            logger.error(f"Failed to load {lang} dataset: {str(e)}")
            xlsum_datasets[lang] = None
    
    return xlsum_datasets

def sample_dataset(dataset, sample_size=2000, random_seed=42):
    """Randomly sample specified number of items from dataset"""
    if dataset is None or len(dataset) == 0:
        return None
    
    random.seed(random_seed)
    total_size = len(dataset)
    
    if total_size <= sample_size:
        return dataset
    
    # Random sampling from list
    indices = random.sample(range(total_size), sample_size)
    sampled_dataset = [dataset[i] for i in indices]
    
    return sampled_dataset

def calculate_metrics(generated_summary: str, reference_summary: str, language: str = "en") -> Dict[str, float]:
    """Calculate ROUGE-1, BLEU, and BERTScore metrics with language-specific settings"""
    if not METRICS_AVAILABLE:
        return {"rouge1": 0.0, "bleu": 0.0, "bertscore": 0.0}
    
    # Language mapping for BERTScore
    bertscore_lang_map = {
        "bengali": "bn",
        "nepali": "ne", 
        "burmese": "my",
        "sinhala": "si",
        "english": "en"
    }
    
    # Get BERTScore language code
    bert_lang = bertscore_lang_map.get(language.lower(), "en")
    
    metrics = {}
    
    try:
        # ROUGE-1 score - works for most languages with proper tokenization
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)  # Disable stemming for non-English
        rouge_scores = scorer.score(reference_summary, generated_summary)
        
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        
        # BLEU score - language agnostic but benefits from proper tokenization
        smoothie = SmoothingFunction().method4
        reference_tokens = [reference_summary.split()]
        generated_tokens = generated_summary.split()
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
        metrics['bleu'] = bleu_score
        
        # BERTScore with language-specific models
        try:
            logger.info(f"Calculating BERTScore for language: {language} (code: {bert_lang})")
            P, R, F1 = bert_score([generated_summary], [reference_summary], lang=bert_lang, verbose=False)
            metrics['bertscore'] = F1.item()  # Use F1 score as the main BERTScore metric
        except Exception as bert_error:
            logger.warning(f"BERTScore calculation failed for {language}: {str(bert_error)}")
            # Fallback to English if language-specific model fails
            if bert_lang != "en":
                try:
                    logger.info(f"Falling back to English BERTScore for {language}")
                    P, R, F1 = bert_score([generated_summary], [reference_summary], lang="en", verbose=False)
                    metrics['bertscore'] = F1.item()
                except Exception as fallback_error:
                    logger.warning(f"English BERTScore fallback also failed: {str(fallback_error)}")
                    metrics['bertscore'] = 0.0
            else:
                metrics['bertscore'] = 0.0
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {language}: {str(e)}")
        metrics = {"rouge1": 0.0, "bleu": 0.0, "bertscore": 0.0}
    
    return metrics

def generate_summary_with_service(text: str, prompt_type: str, service_name: str, language: str = "english") -> str:
    """Generate summary using specified AI service with language-specific prompts"""
    prompt = get_prompt_template(language, prompt_type).format(text=text)
    
    try:
        if service_name == "openai" and openai_service.is_configured():
            return openai_service.chat_completion(prompt)
        elif service_name == "anthropic" and anthropic_service.is_configured():
            return anthropic_service.chat_completion(prompt)
        elif service_name == "deepseek" and deepseek_service.is_configured():
            return deepseek_service.chat_completion(prompt)
        elif service_name == "bloomz" and bloomz_service.is_configured():
            return bloomz_service.generate_text(prompt)
        else:
            return f"Service {service_name} not configured or available"
    except Exception as e:
        logger.error(f"Error generating summary with {service_name}: {str(e)}")
        return f"Error: {str(e)}"

def create_queue_id() -> str:
    """Generate a unique queue ID"""
    return str(uuid.uuid4())[:8]

def save_queue_state(queue_id: str, queue_data: Dict[str, Any]):
    """Save queue state to file"""
    queue_file = f"data/queues/{queue_id}.json"
    with open(queue_file, 'w', encoding='utf-8') as f:
        json.dump(queue_data, f, ensure_ascii=False, indent=2)

def load_queue_state(queue_id: str) -> Dict[str, Any]:
    """Load queue state from file"""
    queue_file = f"data/queues/{queue_id}.json"
    if os.path.exists(queue_file):
        with open(queue_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def update_queue_progress(queue_id: str, progress_data: Dict[str, Any]):
    """Update queue progress"""
    with queue_lock:
        if queue_id in active_queues:
            active_queues[queue_id].update(progress_data)
        
        # Also save to file
        queue_state = load_queue_state(queue_id)
        if queue_state:
            queue_state.update(progress_data)
            save_queue_state(queue_id, queue_state)

def summarization_worker(queue_id: str, samples_file: str, service: str, batch_size: int, 
                        max_tokens: int, resume_batch: int = 0, prompt_type: str = "direct"):
    """Background worker for summarization"""
    try:
        logger.info(f"Starting summarization worker for queue {queue_id}")
        
        # Check if cancelled before starting
        if queue_id in cancelled_queues:
            update_queue_progress(queue_id, {
                'status': 'cancelled',
                'cancelled_at': datetime.now().isoformat()
            })
            return
        
        # Load samples
        df = pd.read_csv(samples_file)
        total_samples = len(df)
        
        # Calculate start position
        start_idx = resume_batch * batch_size
        if start_idx >= total_samples:
            update_queue_progress(queue_id, {
                'status': 'error',
                'error': f'Resume batch {resume_batch} exceeds total batches',
                'completed_at': datetime.now().isoformat()
            })
            return
        
        logger.info(f"Resuming from batch {resume_batch}, starting at index {start_idx}")
        
        # Initialize results
        results = []
        processed_count = 0
        
        # Load existing results if resuming
        results_file = f"data/results/results_{queue_id}.json"
        if resume_batch > 0 and os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_count = len(results)
            logger.info(f"Loaded {processed_count} existing results")
        
        # Process samples
        for i in range(start_idx, total_samples, batch_size):
            # Check for cancellation at the start of each batch
            if queue_id in cancelled_queues:
                logger.info(f"Queue {queue_id} cancelled during processing at batch {i // batch_size}")
                update_queue_progress(queue_id, {
                    'status': 'cancelled',
                    'processed_samples': processed_count,
                    'progress_percentage': (processed_count / total_samples) * 100,
                    'last_batch_completed': (i // batch_size) - 1,
                    'cancelled_at': datetime.now().isoformat()
                })
                return
            
            batch_end = min(i + batch_size, total_samples)
            batch_num = i // batch_size
            
            logger.info(f"Processing batch {batch_num}: samples {i} to {batch_end-1}")
            
            update_queue_progress(queue_id, {
                'status': 'processing',
                'current_batch': batch_num,
                'total_batches': (total_samples + batch_size - 1) // batch_size,
                'processed_samples': processed_count,
                'total_samples': total_samples,
                'progress_percentage': (processed_count / total_samples) * 100
            })
            
            # Process batch
            for idx in range(i, batch_end):
                # Check for cancellation during sample processing
                if queue_id in cancelled_queues:
                    logger.info(f"Queue {queue_id} cancelled during sample {idx}")
                    update_queue_progress(queue_id, {
                        'status': 'cancelled',
                        'processed_samples': processed_count,
                        'progress_percentage': (processed_count / total_samples) * 100,
                        'last_batch_completed': batch_num - 1 if processed_count > 0 else 0,
                        'cancelled_at': datetime.now().isoformat()
                    })
                    return
                
                row = df.iloc[idx]
                
                # Generate summary
                generated_summary = generate_summary_with_service(
                    row['text'], prompt_type, service, row['language']
                )
                
                # Calculate metrics
                metrics = calculate_metrics(generated_summary, row['summary'], row['language'])
                
                # Create result
                result = {
                    "id": f"{row['language']}_{idx}_{service}_{prompt_type}",
                    "original_index": int(idx),
                    "generated_summary": generated_summary,
                    "usual_summary": row['summary'],
                    "scoring_rouge": {"rouge1": metrics['rouge1']},
                    "scoring_bleu": metrics['bleu'],
                    "scoring_bertscore": metrics['bertscore'],
                    "source_url": row.get('url', ''),
                    "title": row.get('title', ''),
                    "content": row['text'],
                    "language": row['language'],
                    "prompt_type": prompt_type,
                    "service": service,
                    "batch_number": batch_num,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                processed_count += 1
                
                # Log progress
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{total_samples} samples")
            
            # Save intermediate results
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Update progress
            update_queue_progress(queue_id, {
                'processed_samples': processed_count,
                'progress_percentage': (processed_count / total_samples) * 100,
                'last_batch_completed': batch_num
            })
        
        # Mark as completed (only if not cancelled)
        if queue_id not in cancelled_queues:
            update_queue_progress(queue_id, {
                'status': 'completed',
                'processed_samples': processed_count,
                'total_samples': total_samples,
                'progress_percentage': 100.0,
                'completed_at': datetime.now().isoformat(),
                'results_file': f"results_{queue_id}.json"
            })
            
            logger.info(f"Summarization completed for queue {queue_id}")
        
    except Exception as e:
        logger.error(f"Error in summarization worker {queue_id}: {str(e)}")
        update_queue_progress(queue_id, {
            'status': 'error',
            'error': str(e),
            'failed_at': datetime.now().isoformat()
        })
    finally:
        # Clean up cancelled queue from tracking set
        if queue_id in cancelled_queues:
            cancelled_queues.discard(queue_id)

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "message": "XLSUM Text Summarization API with Language-Specific Prompts",
        "version": "2.1.0",
        "supported_languages": ["bengali", "nepali", "burmese", "sinhala"],
        "language_support": {
            "bengali": "Native Bengali prompts (বাংলা)",
            "nepali": "Native Nepali prompts (नेपाली)",
            "burmese": "Native Burmese prompts (မြန်မာ)",
            "sinhala": "Native Sinhala prompts (සිංහල)",
            "fallback": "English prompts for unsupported languages"
        },
        "available_services": ["openai", "anthropic", "deepseek", "bloomz"],
        "prompt_types": ["direct", "minimal_details", "analytical_details"],
        "features": [
            "Language-specific summarization prompts",
            "Multi-language evaluation metrics",
            "Queue-based background processing",
            "Resume capability for interrupted jobs",
            "Real-time progress monitoring"
        ],
        "endpoints": {
            "health": "GET /",
            "test_services": "GET /api/test-services",
            "load_datasets": "POST /api/load-datasets",
            "save_sample": "POST /api/save_sample",
            "summarize": "POST /api/summarize",
            "show_progress": "GET /api/show_progress/<queue_id>",
            "stop_queue": "POST /api/stop_queue/<queue_id>",
            "list_queues": "GET /api/list_queues",
            "generate_summaries": "POST /api/generate-summaries",
            "batch_process": "POST /api/batch-process",
            "export_csv": "POST /api/export-csv",
            "dataset_info": "GET /api/dataset-info",
            "download": "GET /download"
        }
    })

@app.route('/api/load-datasets', methods=['POST'])
def load_datasets():
    """Load XLSUM datasets for processing"""
    try:
        datasets = load_xlsum_languages()
        
        # Sample datasets
        for lang, dataset in datasets.items():
            if dataset is not None:
                sampled = sample_dataset(dataset, sample_size=2000)
                processed_datasets[lang] = sampled
                logger.info(f"Sampled {len(sampled)} items for {lang}")
        
        return jsonify({
            "status": "success",
            "message": "Datasets loaded and sampled successfully",
            "datasets": {lang: len(dataset) if dataset else 0 for lang, dataset in processed_datasets.items()}
        })
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_sample', methods=['POST'])
def save_sample():
    """Save sampled data to CSV format"""
    try:
        data = request.get_json()
        languages = data.get('lang', ['bengali', 'nepali', 'burmese', 'sinhala'])
        
        if isinstance(languages, str):
            languages = [languages]
        
        # Validate languages
        valid_languages = ['bengali', 'nepali', 'burmese', 'sinhala']
        invalid_langs = [lang for lang in languages if lang not in valid_languages]
        if invalid_langs:
            return jsonify({"error": f"Invalid languages: {invalid_langs}"}), 400
        
        # Load datasets if not already loaded
        if not xlsum_datasets:
            load_xlsum_languages()
        
        # Sample and save each language
        saved_files = {}
        total_samples = 0
        
        for lang in languages:
            if lang not in xlsum_datasets or xlsum_datasets[lang] is None:
                logger.warning(f"Dataset for {lang} not available")
                continue
            
            # Sample dataset
            sampled_data = sample_dataset(xlsum_datasets[lang], sample_size=2000)
            if sampled_data is None:
                continue
            
            # Convert to DataFrame
            df_data = []
            for i, item in enumerate(sampled_data):
                row = {
                    'index': i,
                    'language': lang,
                    'id': item.get('id', f"{lang}_{i}"),
                    'title': item.get('title', ''),
                    'text': item.get('text', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'text_length': len(item.get('text', '')),
                    'summary_length': len(item.get('summary', '')),
                    'created_at': datetime.now().isoformat()
                }
                df_data.append(row)
            
            # Save to CSV
            df = pd.DataFrame(df_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"samples_{lang}_{timestamp}.csv"
            filepath = f"data/samples/{filename}"
            
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            saved_files[lang] = {
                'filename': filename,
                'filepath': filepath,
                'sample_count': len(df),
                'columns': list(df.columns)
            }
            total_samples += len(df)
            
            logger.info(f"Saved {len(df)} samples for {lang} to {filepath}")
        
        return jsonify({
            "status": "success",
            "message": f"Sampled data saved for {len(saved_files)} languages",
            "languages": languages,
            "saved_files": saved_files,
            "total_samples": total_samples,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saving samples: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-summaries', methods=['POST'])
def generate_summaries():
    """Generate summaries for sampled datasets"""
    try:
        data = request.get_json()
        language = data.get('language', 'bengali')
        prompt_type = data.get('prompt_type', 'direct')
        service = data.get('service', 'openai')
        batch_size = data.get('batch_size', 10)
        start_index = data.get('start_index', 0)
        
        if language not in processed_datasets or processed_datasets[language] is None:
            return jsonify({"error": f"Dataset for {language} not loaded"}), 400
        
        # Validate prompt type - check if it exists in any language template
        valid_prompt_types = ["direct", "minimal_details", "analytical_details"]
        if prompt_type not in valid_prompt_types:
            return jsonify({"error": f"Invalid prompt_type: {prompt_type}. Valid types: {valid_prompt_types}"}), 400
        
        dataset = processed_datasets[language]
        end_index = min(start_index + batch_size, len(dataset))
        
        results = []
        
        for i in range(start_index, end_index):
            item = dataset[i]
            
            # Generate summary
            generated_summary = generate_summary_with_service(
                item['text'], prompt_type, service, language
            )
            
            # Calculate metrics
            metrics = calculate_metrics(generated_summary, item['summary'], language)
            
            # Create result record
            result = {
                "id": f"{language}_{i}_{service}_{prompt_type}",
                "generated_summary": generated_summary,
                "usual_summary": item['summary'],
                "scoring_rouge": {
                    "rouge1": metrics['rouge1']
                },
                "scoring_bleu": metrics['bleu'],
                "scoring_bertscore": metrics['bertscore'],
                "source_url": item.get('url', ''),
                "title": item.get('title', ''),
                "content": item['text'],
                "language": language,
                "prompt_type": prompt_type,
                "service": service,
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
            logger.info(f"Processed item {i+1}/{end_index} for {language}")
        
        return jsonify({
            "status": "success",
            "processed_count": len(results),
            "total_items": len(dataset),
            "start_index": start_index,
            "end_index": end_index,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error generating summaries: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    """Process entire dataset with specified parameters"""
    try:
        data = request.get_json()
        language = data.get('language', 'bengali')
        prompt_type = data.get('prompt_type', 'direct')
        services = data.get('services', ['openai'])
        
        if language not in processed_datasets or processed_datasets[language] is None:
            return jsonify({"error": f"Dataset for {language} not loaded"}), 400
        
        dataset = processed_datasets[language]
        all_results = []
        
        for service in services:
            logger.info(f"Processing {language} dataset with {service} service...")
            
            for i, item in enumerate(dataset):
                # Generate summary
                generated_summary = generate_summary_with_service(
                    item['text'], prompt_type, service, language
                )
                
                # Calculate metrics
                metrics = calculate_metrics(generated_summary, item['summary'], language)
                
                # Create result record
                result = {
                    "id": f"{language}_{i}_{service}_{prompt_type}",
                    "generated_summary": generated_summary,
                    "usual_summary": item['summary'],
                    "scoring_rouge": {
                        "rouge1": metrics['rouge1']
                    },
                    "scoring_bleu": metrics['bleu'],
                    "scoring_bertscore": metrics['bertscore'],
                    "source_url": item.get('url', ''),
                    "title": item.get('title', ''),
                    "content": item['text'],
                    "language": language,
                    "prompt_type": prompt_type,
                    "service": service,
                    "timestamp": datetime.now().isoformat()
                }
                
                all_results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i+1}/{len(dataset)} items with {service}")
        
        # Save results to file
        output_file = f"results_{language}_{prompt_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f"data/{output_file}", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            "status": "success",
            "total_processed": len(all_results),
            "output_file": output_file,
            "summary_stats": {
                "languages": [language],
                "services": services,
                "prompt_type": prompt_type,
                "total_items": len(dataset)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """Export results to CSV format"""
    try:
        data = request.get_json()
        results_file = data.get('results_file')
        
        if not results_file or not os.path.exists(f"data/{results_file}"):
            return jsonify({"error": "Results file not found"}), 400
        
        # Load results
        with open(f"data/{results_file}", 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Convert to DataFrame
        df_data = []
        for result in results:
            row = {
                'id': result['id'],
                'generated_summary': result['generated_summary'],
                'usual_summary': result['usual_summary'],
                'rouge1': result['scoring_rouge']['rouge1'],
                'bleu': result['scoring_bleu'],
                'bertscore': result['scoring_bertscore'],
                'source_url': result['source_url'],
                'title': result['title'],
                'content': result['content'],
                'language': result['language'],
                'prompt_type': result['prompt_type'],
                'service': result['service'],
                'timestamp': result['timestamp']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_file = results_file.replace('.json', '.csv')
        df.to_csv(f"data/{csv_file}", index=False, encoding='utf-8')
        
        return jsonify({
            "status": "success",
            "csv_file": csv_file,
            "total_rows": len(df)
        })
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset-info', methods=['GET'])
def dataset_info():
    """Get information about loaded datasets"""
    info = {}
    
    for lang, dataset in processed_datasets.items():
        if dataset is not None and len(dataset) > 0:
            sample_item = dataset[0]
            info[lang] = {
                "total_samples": len(dataset),
                "sample_item": {
                    "title": (sample_item.get('title', '')[:100] + "...") if len(sample_item.get('title', '')) > 100 else sample_item.get('title', ''),
                    "summary_length": len(sample_item.get('summary', '')),
                    "content_length": len(sample_item.get('text', ''))
                }
            }
        else:
            info[lang] = {"status": "not_loaded"}
    
    return jsonify({
        "datasets": info,
        "available_services": {
            "openai": openai_service.is_configured(),
            "anthropic": anthropic_service.is_configured(),
            "deepseek": deepseek_service.is_configured(),
            "bloomz": bloomz_service.is_configured()
        }
    })

@app.route('/api/test-services', methods=['GET'])
def test_services():
    """Test all AI services to check if they're working"""
    test_prompt = "Summarize this text: The quick brown fox jumps over the lazy dog. This is a simple test sentence."
    
    results = {}
    
    # Test OpenAI
    try:
        if openai_service.is_configured():
            start_time = datetime.now()
            response = openai_service.chat_completion(test_prompt, max_tokens=50)
            end_time = datetime.now()
            
            results['openai'] = {
                'status': 'working' if response and len(response.strip()) > 0 else 'error',
                'response_time': (end_time - start_time).total_seconds(),
                'response_length': len(response) if response else 0,
                'sample_response': response[:100] + "..." if response and len(response) > 100 else response
            }
        else:
            results['openai'] = {'status': 'not_configured', 'error': 'Missing API key'}
    except Exception as e:
        results['openai'] = {'status': 'error', 'error': str(e)}
    
    # Test Anthropic
    try:
        if anthropic_service.is_configured():
            start_time = datetime.now()
            response = anthropic_service.chat_completion(test_prompt, max_tokens=50)
            end_time = datetime.now()
            
            results['anthropic'] = {
                'status': 'working' if response and len(response.strip()) > 0 else 'error',
                'response_time': (end_time - start_time).total_seconds(),
                'response_length': len(response) if response else 0,
                'sample_response': response[:100] + "..." if response and len(response) > 100 else response
            }
        else:
            results['anthropic'] = {'status': 'not_configured', 'error': 'Missing API key'}
    except Exception as e:
        results['anthropic'] = {'status': 'error', 'error': str(e)}
    
    # Test DeepSeek
    try:
        if deepseek_service.is_configured():
            start_time = datetime.now()
            response = deepseek_service.chat_completion(test_prompt, max_tokens=50)
            end_time = datetime.now()
            
            results['deepseek'] = {
                'status': 'working' if response and len(response.strip()) > 0 else 'error',
                'response_time': (end_time - start_time).total_seconds(),
                'response_length': len(response) if response else 0,
                'sample_response': response[:100] + "..." if response and len(response) > 100 else response
            }
        else:
            results['deepseek'] = {'status': 'not_configured', 'error': 'Missing API key'}
    except Exception as e:
        results['deepseek'] = {'status': 'error', 'error': str(e)}
    
    # Test Bloomz
    try:
        if bloomz_service.is_configured():
            start_time = datetime.now()
            response = bloomz_service.generate_text(test_prompt, max_length=50)
            end_time = datetime.now()
            
            results['bloomz'] = {
                'status': 'working' if response and len(response.strip()) > 0 else 'error',
                'response_time': (end_time - start_time).total_seconds(),
                'response_length': len(response) if response else 0,
                'sample_response': response[:100] + "..." if response and len(response) > 100 else response
            }
        else:
            results['bloomz'] = {'status': 'not_configured', 'error': 'Model not loaded'}
    except Exception as e:
        results['bloomz'] = {'status': 'error', 'error': str(e)}
    
    # Calculate summary
    working_count = sum(1 for result in results.values() if result.get('status') == 'working')
    total_count = len(results)
    
    return jsonify({
        'test_prompt': test_prompt,
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'working_services': working_count,
            'total_services': total_count,
            'success_rate': f"{(working_count/total_count)*100:.1f}%"
        },
        'services': results
    })

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Start summarization process in background queue"""
    try:
        data = request.get_json()
        
        # Required parameters
        lang = data.get('lang')
        service = data.get('service', 'openai')
        batch_size = data.get('batch_size', 10)
        max_tokens = data.get('max_token', 7000)
        
        # Optional parameters
        resume_batch = data.get('resume_batch', 0)
        prompt_type = data.get('prompt_type', 'direct')
        samples_file = data.get('samples_file')
        
        # Validation
        if not lang:
            return jsonify({"error": "Language (lang) is required"}), 400
        
        if lang not in ['bengali', 'nepali', 'burmese', 'sinhala']:
            return jsonify({"error": f"Invalid language: {lang}"}), 400
        
        if service not in ['openai', 'anthropic', 'deepseek', 'bloomz']:
            return jsonify({"error": f"Invalid service: {service}"}), 400
        
        # Validate prompt type - check if it exists in any language template
        valid_prompt_types = ["direct", "minimal_details", "analytical_details"]
        if prompt_type not in valid_prompt_types:
            return jsonify({"error": f"Invalid prompt_type: {prompt_type}. Valid types: {valid_prompt_types}"}), 400
        
        # Find samples file if not provided
        if not samples_file:
            # Look for the most recent samples file for this language
            samples_dir = Path("data/samples")
            pattern = f"samples_{lang}_*.csv"
            matching_files = list(samples_dir.glob(pattern))
            
            if not matching_files:
                return jsonify({"error": f"No samples file found for {lang}. Use /api/save_sample first."}), 400
            
            # Get the most recent file
            samples_file = str(max(matching_files, key=os.path.getctime))
        else:
            # Validate provided file path
            if not samples_file.startswith('data/samples/'):
                samples_file = f"data/samples/{samples_file}"
            
            if not os.path.exists(samples_file):
                return jsonify({"error": f"Samples file not found: {samples_file}"}), 400
        
        # Check if service is configured
        service_configured = False
        if service == "openai":
            service_configured = openai_service.is_configured()
        elif service == "anthropic":
            service_configured = anthropic_service.is_configured()
        elif service == "deepseek":
            service_configured = deepseek_service.is_configured()
        elif service == "bloomz":
            service_configured = bloomz_service.is_configured()
        
        if not service_configured:
            return jsonify({"error": f"Service {service} is not configured"}), 400
        
        # Create queue
        queue_id = create_queue_id()
        
        # Initialize queue state
        queue_data = {
            'queue_id': queue_id,
            'status': 'queued',
            'language': lang,
            'service': service,
            'batch_size': batch_size,
            'max_tokens': max_tokens,
            'resume_batch': resume_batch,
            'prompt_type': prompt_type,
            'samples_file': samples_file,
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'processed_samples': 0,
            'total_samples': 0,
            'current_batch': 0,
            'total_batches': 0,
            'progress_percentage': 0.0,
            'error': None
        }
        
        # Save queue state
        save_queue_state(queue_id, queue_data)
        
        # Add to active queues
        with queue_lock:
            active_queues[queue_id] = queue_data.copy()
        
        # Start background worker
        worker_thread = threading.Thread(
            target=summarization_worker,
            args=(queue_id, samples_file, service, batch_size, max_tokens, resume_batch, prompt_type),
            daemon=True
        )
        worker_thread.start()
        
        # Update status to started
        update_queue_progress(queue_id, {
            'status': 'started',
            'started_at': datetime.now().isoformat()
        })
        
        logger.info(f"Started summarization queue {queue_id} for {lang} using {service}")
        
        return jsonify({
            "status": "success",
            "message": "Summarization started",
            "queue_id": queue_id,
            "language": lang,
            "service": service,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "resume_batch": resume_batch,
            "prompt_type": prompt_type,
            "samples_file": samples_file,
            "created_at": queue_data['created_at']
        })
        
    except Exception as e:
        logger.error(f"Error starting summarization: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/show_progress/<queue_id>', methods=['GET'])
def show_progress(queue_id: str):
    """Show progress of summarization queue"""
    try:
        # First check active queues
        with queue_lock:
            if queue_id in active_queues:
                queue_data = active_queues[queue_id].copy()
            else:
                # Load from file
                queue_data = load_queue_state(queue_id)
        
        if not queue_data:
            return jsonify({"error": f"Queue {queue_id} not found"}), 404
        
        # Calculate additional metrics
        if queue_data.get('total_samples', 0) > 0:
            progress_percentage = (queue_data.get('processed_samples', 0) / queue_data['total_samples']) * 100
            queue_data['progress_percentage'] = round(progress_percentage, 2)
        
        # Add estimated time remaining if processing
        if queue_data.get('status') == 'processing' and queue_data.get('started_at'):
            try:
                started_time = datetime.fromisoformat(queue_data['started_at'])
                elapsed_time = (datetime.now() - started_time).total_seconds()
                processed = queue_data.get('processed_samples', 0)
                total = queue_data.get('total_samples', 0)
                
                if processed > 0 and total > processed:
                    avg_time_per_sample = elapsed_time / processed
                    remaining_samples = total - processed
                    estimated_remaining = avg_time_per_sample * remaining_samples
                    
                    queue_data['elapsed_time_seconds'] = round(elapsed_time, 2)
                    queue_data['estimated_remaining_seconds'] = round(estimated_remaining, 2)
                    queue_data['estimated_completion'] = (datetime.now() + 
                                                        pd.Timedelta(seconds=estimated_remaining)).isoformat()
            except Exception:
                pass  # Skip time estimation if there's an error
        
        # Add file information
        if queue_data.get('status') == 'completed':
            results_file = f"data/results/results_{queue_id}.json"
            if os.path.exists(results_file):
                queue_data['results_file_size'] = os.path.getsize(results_file)
                queue_data['results_available'] = True
            else:
                queue_data['results_available'] = False
        
        return jsonify({
            "queue_id": queue_id,
            "queue_data": queue_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting progress for queue {queue_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/list_queues', methods=['GET'])
def list_queues():
    """List all queues"""
    try:
        queues = []
        
        # Get active queues
        with queue_lock:
            for queue_id, queue_data in active_queues.items():
                queues.append({
                    'queue_id': queue_id,
                    'status': queue_data.get('status', 'unknown'),
                    'language': queue_data.get('language', ''),
                    'service': queue_data.get('service', ''),
                    'created_at': queue_data.get('created_at', ''),
                    'progress_percentage': queue_data.get('progress_percentage', 0),
                    'source': 'active'
                })
        
        # Get queues from files
        queue_dir = Path("data/queues")
        if queue_dir.exists():
            for queue_file in queue_dir.glob("*.json"):
                queue_id = queue_file.stem
                if queue_id not in active_queues:  # Don't duplicate active queues
                    try:
                        queue_data = load_queue_state(queue_id)
                        if queue_data:
                            queues.append({
                                'queue_id': queue_id,
                                'status': queue_data.get('status', 'unknown'),
                                'language': queue_data.get('language', ''),
                                'service': queue_data.get('service', ''),
                                'created_at': queue_data.get('created_at', ''),
                                'progress_percentage': queue_data.get('progress_percentage', 0),
                                'source': 'file'
                            })
                    except Exception:
                        continue  # Skip corrupted queue files
        
        # Sort by creation time (newest first)
        queues.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            "status": "success",
            "total_queues": len(queues),
            "queues": queues,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error listing queues: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop_queue/<queue_id>', methods=['POST'])
def stop_queue(queue_id: str):
    """Stop/cancel a running queue"""
    try:
        # Check if queue exists
        queue_data = None
        
        # First check active queues
        with queue_lock:
            if queue_id in active_queues:
                queue_data = active_queues[queue_id].copy()
            else:
                # Load from file
                queue_data = load_queue_state(queue_id)
        
        if not queue_data:
            return jsonify({"error": f"Queue {queue_id} not found"}), 404
        
        current_status = queue_data.get('status', 'unknown')
        
        # Check if queue can be cancelled
        if current_status in ['completed', 'error', 'cancelled']:
            return jsonify({
                "error": f"Cannot stop queue {queue_id}. Current status: {current_status}",
                "current_status": current_status
            }), 400
        
        # Add to cancelled queues set
        cancelled_queues.add(queue_id)
        
        # Update queue status
        if current_status in ['queued', 'started']:
            # If not yet processing, mark as cancelled immediately
            update_queue_progress(queue_id, {
                'status': 'cancelled',
                'cancelled_at': datetime.now().isoformat(),
                'cancellation_reason': 'User requested cancellation'
            })
            
            # Remove from active queues if present
            with queue_lock:
                if queue_id in active_queues:
                    del active_queues[queue_id]
            
            logger.info(f"Queue {queue_id} cancelled immediately (status: {current_status})")
            
            return jsonify({
                "status": "success",
                "message": f"Queue {queue_id} cancelled successfully",
                "queue_id": queue_id,
                "previous_status": current_status,
                "new_status": "cancelled",
                "cancelled_at": datetime.now().isoformat()
            })
        
        elif current_status == 'processing':
            # If currently processing, mark for cancellation (worker will handle it)
            logger.info(f"Queue {queue_id} marked for cancellation (currently processing)")
            
            return jsonify({
                "status": "success",
                "message": f"Queue {queue_id} marked for cancellation. It will stop after the current batch.",
                "queue_id": queue_id,
                "previous_status": current_status,
                "note": "The queue will stop gracefully after completing the current batch",
                "cancellation_requested_at": datetime.now().isoformat()
            })
        
        else:
            return jsonify({
                "error": f"Unknown queue status: {current_status}",
                "queue_id": queue_id,
                "current_status": current_status
            }), 400
        
    except Exception as e:
        logger.error(f"Error stopping queue {queue_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download', methods=['GET'])
def download_data():
    """Download all files from the data folder as a zip file"""
    try:
        data_dir = Path("data")
        
        # Check if data directory exists
        if not data_dir.exists():
            return jsonify({"error": "Data directory not found"}), 404
        
        # Check if data directory has any files
        all_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        if not all_files:
            return jsonify({"error": "No files found in data directory"}), 404
        
        # Create a temporary zip file
        temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"xlsum_data_{timestamp}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        logger.info(f"Creating zip file: {zip_path}")
        
        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in all_files:
                # Get relative path from data directory
                relative_path = os.path.relpath(file_path, start=".")
                zipf.write(file_path, relative_path)
                logger.debug(f"Added to zip: {relative_path}")
        
        # Get zip file size for logging
        zip_size = os.path.getsize(zip_path)
        logger.info(f"Zip file created successfully: {zip_filename} ({zip_size} bytes)")
        
        # Send the file and clean up after
        def remove_temp_file():
            """Clean up temporary files after sending"""
            try:
                time.sleep(1)  # Give time for file to be sent
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")
        
        # Schedule cleanup in background
        cleanup_thread = threading.Thread(target=remove_temp_file, daemon=True)
        cleanup_thread.start()
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Error creating download zip: {str(e)}")
        return jsonify({"error": f"Failed to create download: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 