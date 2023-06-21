
for id in 1 2 3 4 5
do
python3 test.py \
--model_path checkpoints/split${id}/model.pth \
--device cuda:1 \
--data split${id}
done
