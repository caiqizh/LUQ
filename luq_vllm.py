
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
import numpy as np
import os
import time 
import os
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pytz
import argparse

class LUQ_vllm:

    def __init__(
        self,
        model = "llama3-8b-instruct",
        method = "binary",
        abridged = False,
    ):
        """
        model: str
            The model to use. Currently only "llama3-8b-instruct" is supported. If you want to use other more lightweight models, please revise the codes accordingly.
        method: str
            The method to use. Currently only "binary" and "multiclass" are supported. We recommend using "binary" for simplicity.
        abridged: bool
            To have some results quicklier. If True, the function will return the score of the first sentence only. The score then represents the model's confidence in the first sentence given a fixed prompt. 
        """

        if model == "llama3-8b-instruct":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        else:
            raise ValueError("Model not supported")

        self.method = method
        self.abridged = abridged
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Define sampling parameters
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=0.9,
            max_tokens=5,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True,
        )

        self.llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.5)
        
        if self.method == "binary":
            self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
            self.text_mapping = {'yes': 1, 'no': 0, 'n/a': 0.5}
        elif self.method == "multiclass":
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence supported, refuted or not mentioned by the context above? "
                "You should answer the question purely based on the given context. "
                "Do not output the explanations.\n\n"
                "Your answer should be within \"supported\", \"refuted\", or \"not mentioned\".\n\n"
                "Answer: "
            )
            self.text_mapping = {'supported': 1, 'refuted': 0, 'not mentioned': -1, 'n/a': 0.5}

        self.not_defined_text = set()


    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def completion(self, prompts: str):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return outputs

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[List[str]],
        verbose: bool = False,
    ):
        all_samples = [sentences] + sampled_passages

        luq_scores = np.zeros(len(all_samples))
        for index, item in enumerate(all_samples):
            samples = [" ".join(sample) for sample in all_samples if sample != item]

            num_sentences = len(item)
            num_samples = len(samples)
            scores = np.zeros((num_sentences, num_samples))
            
            for sent_i in range(num_sentences):
                prompts = []
                sentence = item[sent_i]
                for sample_i, sample in enumerate(samples):
                    sample = sample.replace("\n", " ") 
                    prompt_text = self.prompt_template.format(context=sample, sentence=sentence)

                    # print(prompt_text)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text}
                    ]

                    prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                    prompts.append(prompt)

                outputs = self.completion(prompts)

                for sample_i, output in enumerate(outputs):
                    generate_text = output.outputs[0].text
                    # print(generate_text)
                    score_ = self.text_postprocessing(generate_text)
                    # print(score_)
                    scores[sent_i, sample_i] = score_

            scores_filtered = np.ma.masked_equal(scores, -1)
            scores_per_sentence = scores_filtered.mean(axis=-1)
            scores_per_sentence = np.where(scores_per_sentence.mask, 0, scores_per_sentence)
            # Calculate the average score for each sentence
            luq_scores[index] = scores_per_sentence.mean()
            print(scores_per_sentence)
            if self.abridged:
                return scores_per_sentence.mean()
        return luq_scores.mean()
        

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        """

        if self.method == "binary":
            text = text.lower().strip()
            if text[:3] == 'yes':
                text = 'yes'
            elif text[:2] == 'no':
                text = 'no'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
        
        elif self.method == "multiclass":
            text = text.lower().strip()
            if text[:7] == 'support':
                text = 'supported'
            elif text[:5] == 'refut':
                text = 'refuted'
            elif text[:3] == 'not':
                text = 'not mentioned'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
            # return text

if __name__ == "__main__":

    LUQ_vllm = LUQ_vllm(model = "llama3-8b-instruct", method = "binary", abridged = True)

    sentences = ['Michael Alan Weiner (born March 31, 1942) is an American radio host.', 'He is the host of The Savage Nation.', "He was a good student when he was in American."]

    # Other samples generated by the same LLM to perform self-check for consistency. The samples can be a single string or a list of facts.
    sample1 = ["Michael Alan Weiner (born March 31, 1942) is an American radio host.", "He is the host of The Savage Country."]
    sample2 = ["Michael Alan Weiner (born January 13, 1960) is a Canadian radio host.", "He works at The New York Times."]
    sample3 = ["Michael Alan Weiner (born March 31, 1942) is an American radio host", "He obtained his PhD from MIT."]

    luq_scores = LUQ_vllm.predict(sentences, [sample1, sample2, sample3])

    print(luq_scores)
