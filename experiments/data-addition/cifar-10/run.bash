rm -r analyses
rm -r models
python pretrain.py
python analyze.py --factor-strategy ekfac
python analyze.py --factor-strategy identity
python train_candidates.py --factor-strategy ekfac
python train_candidates.py --factor-strategy identity
python train_candidates.py --factor-strategy random