# cnn_dailymail GPT-J test, MLPerf v3.1 compliant 
To use MLPerf GPT-J model
```
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download --output-document checkpoint.zip
unzip checkpoint.zip
```
```
python examples/run_mlperf_cnn_dailymail.py \
  --model checkpoint \
  --tasks cnn_dailymail \
  --batch_size 1 \
  --output_path ./output \
  --device cpu
```

# google/flan-t5-xl lambada test, according to https://github.com/EleutherAI/lm-evaluation-harness/issues/1017 skip eos when computing metrics 
```
python examples/run_flan-t5-xl_lambada.py \
  --model google/flan-t5-xl \
  --tasks lambada_openai \
  --batch_size 1 \
  --output_path ./output \
  --device cpu
```

# run lambada test with next token on greedy search
python examples/run_lambada_greedy.py \
  --model EleutherAI/gpt-j-6b \
  --tasks lambada_openai_greedy \
  --batch_size 1 \
  --output_path ./output \
  --device cpu


