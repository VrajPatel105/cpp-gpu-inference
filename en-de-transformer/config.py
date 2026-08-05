# just giving default numbers that are generally used 
configurations = {
    'd_model' : 512,
    'num_heads' : 8,
    'num_blocks' : 6,
    'src_max_seq_len' : 323,
    'tgt_max_seq_len' : 123,
    'src_vocab_size' : 10000,
    'tgt_vocab_size' : 3883, # based on our german dict
    'path' : '/mnt/c/dev/projects/cpp-gpu-inference/en-de-transformer/English-German.tsv',
    'max_len': 100,
    'batch_size' : 32,
    'learning_rate' : 0.0001,
    'epochs' : 10
}