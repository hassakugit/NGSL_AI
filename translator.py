# --- translator.py ---
import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import spacy
from spacy.lang.ja import Japanese

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Translator:
    """Handles Japanese sentence splitting, translation, and vocabulary constraint rewriting."""

    def __init__(self, translation_model_name: str, causal_lm_model_name: str, device: str, vocab_dir: str, trust_remote_code: bool, max_rewrite_attempts: int):
        """
        Initializes the Translator, loading models and vocabulary.

        Args:
            translation_model_name: Hugging Face model ID for Seq2Seq translation.
            causal_lm_model_name: Hugging Face model ID for Causal LM rewriting.
            device: The device to run models on ('cuda', 'mps', 'cpu').
            vocab_dir: Path to the directory containing 'weekN.txt' vocabulary files.
            trust_remote_code: Whether to trust remote code execution for models.
            max_rewrite_attempts: Maximum iterations for vocabulary rewriting loop.
        """
        self.device_name = device
        self.device = torch.device(self.device_name)
        self.vocab_dir = Path(vocab_dir)
        self.max_rewrite_attempts = max_rewrite_attempts
        self.trust_remote_code = trust_remote_code

        logger.info(f"Using device: {self.device_name}")

        # Load Japanese NLP model for sentence splitting
        # Using spaCy with a simple Japanese model. Consider SudachiPy for more complex needs.
        try:
            # Check if model is installed, if not, download might fail here depending on permissions
            # Best practice is to ensure download in Dockerfile
            self.nlp = spacy.load("ja_core_news_sm")
            # Add sentence segmenter if not already present by default for the model
            if not self.nlp.has_pipe("sentencizer"):
                 self.nlp.add_pipe("sentencizer")
            logger.info("Japanese NLP model (spaCy) loaded successfully.")
        except OSError:
            logger.error("Could not load spaCy Japanese model 'ja_core_news_sm'.")
            logger.error("Please ensure it's downloaded. Try: python -m spacy download ja_core_news_sm")
            # Fallback to basic punctuation splitting if spaCy fails
            self.nlp = None
            logger.warning("Falling back to basic punctuation-based sentence splitting.")


        # Load Translation Model
        logger.info(f"Loading translation model: {translation_model_name}...")
        try:
            # Explicitly load tokenizer and model for more control if needed
            # self.translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
            # self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to(self.device)
            # Using pipeline for simplicity
            self.translation_pipeline = pipeline(
                "translation",
                model=translation_model_name,
                device=self.device_map(self.device_name), # pipeline expects -1 for cpu, 0 for cuda:0 etc.
                trust_remote_code=self.trust_remote_code # Some models might need this
            )
            logger.info("Translation model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading translation model {translation_model_name}: {e}", exc_info=True)
            raise

        # Load Causal LM for Rewriting
        logger.info(f"Loading Causal LM: {causal_lm_model_name}...")
        try:
            # Using pipeline for generation tasks
            self.causal_lm_pipeline = pipeline(
                "text-generation",
                model=causal_lm_model_name,
                tokenizer=AutoTokenizer.from_pretrained(causal_lm_model_name), # Ensure tokenizer is loaded
                device=self.device_map(self.device_name),
                torch_dtype=torch.bfloat16 if self.device_name != 'cpu' else torch.float32, # Use bfloat16 on GPU/MPS if available
                trust_remote_code=self.trust_remote_code # Crucial for models like Phi-3
            )
            # Set pad token if not set (common issue for generation with some models)
            if self.causal_lm_pipeline.tokenizer.pad_token_id is None:
                self.causal_lm_pipeline.tokenizer.pad_token = self.causal_lm_pipeline.tokenizer.eos_token
                self.causal_lm_pipeline.model.config.pad_token_id = self.causal_lm_pipeline.model.config.eos_token_id

            logger.info("Causal LM loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Causal LM {causal_lm_model_name}: {e}", exc_info=True)
            raise

        # Load Vocabulary Lists
        self.vocab_lists = self._load_vocab_lists()
        if not self.vocab_lists:
            logger.warning(f"No vocabulary lists found or loaded from {self.vocab_dir}. Vocabulary check will be ineffective.")


    def device_map(self, device_name: str) -> int:
        """Maps device name string ('cuda', 'mps', 'cpu') to pipeline device index."""
        if device_name == "cuda" and torch.cuda.is_available():
            return 0 # Use the first CUDA device
        elif device_name == "mps" and torch.backends.mps.is_available():
             # Pipelines might not directly support 'mps', often fallback needed or use device mapping features
             # Returning 0 might work if transformers maps it internally, -1 forces CPU
             # Let's try returning -1 and rely on the main device setting for MPS if pipeline doesn't handle it well.
             # For now, let's force CPU for pipeline if MPS is selected, as direct support varies.
             # UPDATE: Newer transformers/accelerate might handle 'mps' better directly in pipeline. Try passing device object?
             # Let's try device index mapping. MPS is often index 0 when available but not CUDA.
             # Safest for pipeline seems to be CPU if not CUDA.
             # return 0 # Try this if pipeline supports MPS index
             return -1 # Force CPU for pipeline if MPS selected, rely on direct model.to(device) for MPS
        else: # cpu or fallback
            return -1


    def _load_vocab_lists(self) -> Dict[int, Set[str]]:
        """Loads vocabulary files (weekN.txt) from the specified directory."""
        vocab = {}
        if not self.vocab_dir.is_dir():
            logger.error(f"Vocabulary directory not found: {self.vocab_dir}")
            return vocab

        for filepath in self.vocab_dir.glob("week*.txt"):
            match = re.match(r"week(\d+)\.txt", filepath.name)
            if match:
                week_num = int(match.group(1))
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        # Read words, strip whitespace, convert to lowercase, ignore empty lines
                        words = {line.strip().lower() for line in f if line.strip()}
                        vocab[week_num] = words
                        logger.info(f"Loaded {len(words)} words for week {week_num} from {filepath.name}")
                except Exception as e:
                    logger.error(f"Error reading vocabulary file {filepath.name}: {e}")
        logger.info(f"Loaded vocabulary for weeks: {sorted(vocab.keys())}")
        return vocab

    def _get_target_vocab(self, week: int) -> Set[str]:
        """Retrieves the vocabulary set for the target week."""
        # Currently returns only the specific week's list.
        # TODO: Consider an option to combine lists up to the target week.
        return self.vocab_lists.get(week, set())

    def _split_sentences(self, text: str) -> List[str]:
        """Splits Japanese text into sentences."""
        if self.nlp:
            try:
                doc = self.nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception as e:
                logger.warning(f"spaCy sentence splitting failed: {e}. Falling back to basic splitting.")
                # Fallback if spacy fails for some reason
                return self._basic_split_sentences(text)
        else:
            # Use basic fallback if spaCy wasn't loaded
            return self._basic_split_sentences(text)

    def _basic_split_sentences(self, text: str) -> List[str]:
        """A very basic sentence splitter based on punctuation."""
        text = re.sub(r"([。？！])", r"\1\n", text) # Add newline after standard Japanese terminators
        text = re.sub(r"(\.)\s+", r"\1\n", text)    # Add newline after period+space (for potential mixed input)
        sentences = [s.strip() for s in text.splitlines() if s.strip()]
        logger.debug(f"Basic split sentences: {sentences}")
        return sentences

    def _initial_translate(self, japanese_sentences: List[str]) -> List[str]:
        """Translates a list of Japanese sentences to English using the Seq2Seq model."""
        if not japanese_sentences:
            return []
        try:
            # Batch translation is generally more efficient
            translations = self.translation_pipeline(japanese_sentences)
            # Ensure output is a list of strings
            english_sentences = [t['translation_text'] for t in translations]
            logger.info(f"Initial translation completed for {len(japanese_sentences)} sentences.")
            logger.debug(f"Translations: {english_sentences}")
            return english_sentences
        except Exception as e:
            logger.error(f"Error during initial translation: {e}", exc_info=True)
            # Return empty strings or handle error appropriately
            return ["<translation error>"] * len(japanese_sentences)

    def _tokenize_english(self, text: str) -> List[str]:
        """Simple English tokenizer: lowercase and split by non-alphanumeric chars."""
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _calculate_coverage(self, text: str, target_vocab: Set[str]) -> Tuple[float, List[str]]:
        """Calculates vocabulary coverage and identifies violating words."""
        if not target_vocab:
            return 100.0, [] # No restrictions if no vocab provided

        words = self._tokenize_english(text)
        if not words:
            return 100.0, [] # Empty text has full coverage

        in_vocab_count = 0
        violations = []
        for word in words:
            if word in target_vocab:
                in_vocab_count += 1
            else:
                violations.append(word)

        coverage = (in_vocab_count / len(words)) * 100.0
        unique_violations = sorted(list(set(violations)))
        return coverage, unique_violations

    def _build_rewrite_prompt(self, original_text: str, target_vocab_list: List[str], violations: List[str]) -> str:
        """Constructs the prompt for the Causal LM to rewrite text."""
        # This prompt structure is crucial and may need tuning based on the Causal LM used.
        # Using a structure common for instruction-following models like Phi-3.

        # Limiting the vocabulary list size in the prompt if it's huge
        max_vocab_in_prompt = 200
        display_vocab = sorted(target_vocab_list)
        if len(display_vocab) > max_vocab_in_prompt:
             # Prioritize showing some common words and perhaps some less common ones?
             # Simple truncation for now:
             display_vocab = display_vocab[:max_vocab_in_prompt] + ["... (list truncated)"]

        prompt = f"""<|user|>
Rewrite the following English text to ensure it primarily uses words from the 'Allowed Vocabulary'.
Avoid using the 'Forbidden Words'.
Preserve the original meaning as closely as possible. Be concise and natural.

Allowed Vocabulary:
{', '.join(display_vocab)}

Forbidden Words:
{', '.join(sorted(list(set(violations))))}

Original Text:
{original_text}

<|end|>
<|assistant|>
Rewritten Text:
"""
        # Note: The specific tokens like <|user|>, <|end|>, <|assistant|> are common for models like Phi-3.
        # Adjust these based on the specific Causal LM's required prompt format.
        return prompt

    def _rewrite_with_causal_lm(self, text_to_rewrite: str, target_vocab: Set[str], current_violations: List[str]) -> str:
        """Uses the Causal LM to rewrite the text based on vocabulary constraints."""
        if not self.causal_lm_pipeline or not target_vocab:
            logger.warning("Cannot rewrite: Causal LM not loaded or target vocabulary is empty.")
            return text_to_rewrite # Return original if rewrite cannot be performed

        prompt = self._build_rewrite_prompt(text_to_rewrite, list(target_vocab), current_violations)
        logger.debug(f"Rewrite Prompt:\n{prompt}")

        try:
            # Configure generation parameters
            # max_length needs to be sufficient for the rewritten text.
            # max_new_tokens might be better to control output length relative to input.
            # Experiment with temperature, top_p, etc. for desired output style.
            # Make sure to handle the prompt format correctly (e.g., stopping at <|end|>)
            outputs = self.causal_lm_pipeline(
                prompt,
                max_new_tokens=len(text_to_rewrite.split()) + 50, # Estimate output length
                num_return_sequences=1,
                do_sample=True, # Use sampling for potentially more natural rewrites
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.causal_lm_pipeline.tokenizer.eos_token_id, # Set pad_token_id
                # Stop sequences might be needed depending on the model's fine-tuning
                # eos_token_id=self.causal_lm_pipeline.tokenizer.eos_token_id
            )

            # Extract the generated text after the prompt's assistant marker
            raw_generated_text = outputs[0]['generated_text']
            logger.debug(f"Raw Causal LM Output:\n{raw_generated_text}")

            # Extract text after the assistant marker
            assistant_marker = "<|assistant|>\nRewritten Text:\n"
            if assistant_marker in raw_generated_text:
                rewritten_text = raw_generated_text.split(assistant_marker, 1)[-1].strip()
                # Further cleanup if the model includes eos tokens or other artifacts
                rewritten_text = rewritten_text.replace(self.causal_lm_pipeline.tokenizer.eos_token, "").strip()
                logger.info("Text rewritten by Causal LM.")
                logger.debug(f"Cleaned Rewritten Text:\n{rewritten_text}")
                return rewritten_text
            else:
                logger.warning("Could not find assistant marker in Causal LM output. Returning raw output segment.")
                # Fallback: try to return something sensible if marker missing
                return raw_generated_text # Or maybe return original? Let's return raw for debug

        except Exception as e:
            logger.error(f"Error during Causal LM rewriting: {e}", exc_info=True)
            return text_to_rewrite # Return original text on error

    def translate(self, japanese_text: str, week: int, threshold: float) -> Dict:
        """
        Performs the full translation process: split, translate, check/rewrite.

        Args:
            japanese_text: The input Japanese text.
            week: The target NGSL week number for vocabulary.
            threshold: The desired minimum vocabulary coverage percentage (0-100).

        Returns:
            A dictionary containing:
            - 'original_japanese': The input Japanese text.
            - 'simplified_japanese': List of sentences after splitting.
            - 'initial_english': English translation before vocabulary processing.
            - 'final_english': English translation after vocabulary processing.
            - 'coverage': Final vocabulary coverage percentage.
            - 'violations': List of unique words in the final translation not in the target vocabulary.
        """
        logger.info(f"Starting translation for week {week}, threshold {threshold}%")
        logger.debug(f"Input Japanese:\n{japanese_text}")

        # 1. Split Japanese text
        simplified_japanese_sentences = self._split_sentences(japanese_text)
        logger.debug(f"Simplified Japanese sentences: {simplified_japanese_sentences}")

        # 2. Initial Translation
        initial_english_sentences = self._initial_translate(simplified_japanese_sentences)
        initial_english_full = " ".join(initial_english_sentences)
        logger.debug(f"Initial English translation:\n{initial_english_full}")

        # 3. Vocabulary Check and Rewrite Loop
        target_vocab = self._get_target_vocab(week)
        if not target_vocab:
            logger.warning(f"No vocabulary list found for week {week}. Skipping vocabulary check.")
            final_english = initial_english_full
            coverage, violations = self._calculate_coverage(final_english, target_vocab) # Should be 100% if vocab is empty
        else:
            logger.info(f"Target vocabulary loaded: {len(target_vocab)} words for week {week}.")
            current_english = initial_english_full
            final_english = initial_english_full # Initialize final_english
            coverage = 0.0
            violations = []

            for attempt in range(self.max_rewrite_attempts):
                logger.info(f"Rewrite attempt {attempt + 1}/{self.max_rewrite_attempts}")
                coverage, violations = self._calculate_coverage(current_english, target_vocab)
                logger.info(f"Attempt {attempt + 1}: Coverage = {coverage:.2f}%, Violations = {len(violations)}")
                logger.debug(f"Violating words: {violations}")

                if coverage >= threshold:
                    logger.info(f"Threshold ({threshold}%) met or exceeded.")
                    final_english = current_english
                    break # Exit loop if threshold is met

                if not violations:
                     logger.info("No violations found, but threshold not met (should not happen unless threshold > 100?). Stopping.")
                     final_english = current_english
                     break

                logger.info("Threshold not met. Rewriting text...")
                current_english = self._rewrite_with_causal_lm(current_english, target_vocab, violations)

                if attempt == self.max_rewrite_attempts - 1:
                    # Last attempt, calculate final coverage and store result
                    final_english = current_english
                    coverage, violations = self._calculate_coverage(final_english, target_vocab)
                    logger.warning(f"Max rewrite attempts reached. Final coverage: {coverage:.2f}%")
                    break # Exit loop

            # Ensure final values are set even if loop didn't run (e.g., threshold 0)
            if self.max_rewrite_attempts == 0 or not target_vocab:
                 final_english = initial_english_full
                 coverage, violations = self._calculate_coverage(final_english, target_vocab)


        return {
            "original_japanese": japanese_text,
            "simplified_japanese": simplified_japanese_sentences,
            "initial_english": initial_english_full,
            "final_english": final_english,
            "coverage": round(coverage, 2),
            "violations": violations,
        }
