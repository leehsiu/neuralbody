docker run -it --rm --gpus=all --runtime nvidia \
--user "$(id -u):$(id -g)" \
--mount type=bind,source="$(pwd)",target=/app \
--mount type=bind,source=<WEIGHT_PATH>,target=/app/weights \
--mount type=bind,source=<MOCAP_DATA_PATH>,target=/app/mocap \
neuralbody python render.py --cfg_file configs/zju_mocap_exp/latent_xyzc_313.yaml exp_name xyzc_313 test_novel_pose True
