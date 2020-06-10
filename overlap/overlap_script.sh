python generate_test_train_data.py
python overlap_train.py --num_units 100 --train_vggish=True --checkpoint ../vggish_model.ckpt
python overlap_train.py --num_units 200 --train_vggish=True --checkpoint ../vggish_model.ckpt
python overlap_train.py --num_units 400 --train_vggish=True --checkpoint ../vggish_model.ckpt
