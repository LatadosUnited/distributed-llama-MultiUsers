#!/bin/sh

./dllama chat --model models\llama3_2_1b_instruct_q40\dllama_model_llama3_2_1b_instruct_q40.m --tokenizer models\llama3_2_1b_instruct_q40\dllama_tokenizer_llama3_2_1b_instruct_q40.t --buffer-float-type q80 --nthreads 4 --max-seq-len 4096
