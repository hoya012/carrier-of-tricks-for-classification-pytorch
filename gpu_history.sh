 python main.py --checkpoint_name baseline_scratch --pretrained 0;
 python main.py --checkpoint_name baseline;
 python main.py --checkpoint_name baseline_warmup --decay_type step_warmup;
 python main.py --checkpoint_name baseline_zerogamma --zero_gamma ;
 python main.py --checkpoint_name baseline_warmup_zerogamma --decay_type step_warmup --zero_gamma;

 python main.py --checkpoint_name baseline_Adam --optimizer ADAM --learning_rate 0.0001
 python main.py --checkpoint_name baseline_Adam_warmup --optimizer ADAM --learning_rate 0.0001 --decay_type step_warmup;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup;
 python main.py --checkpoint_name baseline_Adam_warmup_labelsmooth --optimizer ADAM --learning_rate 0.0001 --decay_type step_warmup --label_smooth 0.1;
 python main.py --checkpoint_name baseline_Adam_warmup_mixup --optimizer ADAM --learning_rate 0.0001 --decay_type step_warmup --mixup 0.2;
 python main.py --checkpoint_name baseline_Adam_warmup_labelsmooth_mixup --optimizer ADAM --learning_rate 0.0001 --decay_type step_warmup --label_smooth 0.1 --mixup 0.2;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine_labelsmooth --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine_mixup --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --mixup 0.2;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine_labelsmooth_mixup --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1 --mixup 0.2;

 python main.py --checkpoint_name baseline_Adam_warmup_cosine_cutmix --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;
 python main.py --checkpoint_name baseline_RAdam_warmup_cosine_labelsmooth --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine_labelsmooth_randaug --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1 --randaugment;
 python main.py --checkpoint_name baseline_Adam_warmup_cosine_labelsmmoth_evonorm --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1 --norm evonorm;
 python main.py --checkpoint_name baseline_RAdam_warmup_cosine_cutmix --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;

 python main.py --checkpoint_name efficientnet_Adam_warmup_cosine_labelsmooth --model EfficientNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name efficientnet_Adam_warmup_cosine_labelsmooth_mixup --model EfficientNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1 --mixup 0.2;
 python main.py --checkpoint_name efficientnet_Adam_warmup_cosine_cutmix --model EfficientNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;
 python main.py --checkpoint_name efficientnet_RAdam_warmup_cosine_labelsmooth --model EfficientNet --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name efficientnet_RAdam_warmup_cosine_cutmix --model EfficientNet --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;

 python main.py --checkpoint_name regnet_Adam_warmup_cosine_labelsmooth --model RegNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name regnet_Adam_warmup_cosine_labelsmooth_mixup --model RegNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1 --mixup 0.2;
 python main.py --checkpoint_name regnet_Adam_warmup_cosine_cutmix --model RegNet --optimizer ADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;
 python main.py --checkpoint_name regnet_RAdam_warmup_cosine_labelsmooth --model RegNet --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --label_smooth 0.1;
 python main.py --checkpoint_name regnet_RAdam_warmup_cosine_cutmix --model RegNet --optimizer RADAM --learning_rate 0.0001 --decay_type cosine_warmup --cutmix_alpha 1.0 --cutmix_prob 1.0;