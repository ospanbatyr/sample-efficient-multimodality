export TOKENIZERS_PARALLELISM=true

# projector pre-training
python -u train_projector,py configs/projector/v1:llama1b_inst_all_extracted.json ;

# hypernetwork training
python -u train_hypernet.py configs/hypernet/v4:llama1b_inst_all.json ;

# adaptation
python -u train_hypernet.py configs/hypernet/v6:llama1b_inst_all_only_fewshot_chebi20_32.json ;
python -u train_lora.py configs/lora/v3:llama1b_inst_mlp2_chebi20_32.json ;
python -u train_projector.py configs/projector/chebi20/v2:llama1b_chebi20_mlp2_32_ft.json ;
python -u train_projector.py configs/projector/chebi20/v2:llama1b_chebi20_mlp2_32.json ;
#  ...