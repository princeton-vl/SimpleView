# evaluation on test set, fair protocol, table 4
python main.py --entry test --model-path pretrained/dgcnn_rscnn_run_1/model_300.pth --exp-config configs/dgcnn_rscnn_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_dgcnn_run_1/model_325.pth --exp-config configs/dgcnn_dgcnn_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet_run_1/model_300.pth --exp-config configs/dgcnn_pointnet_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet2_run_1/model_975.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_simpleview_run_1/model_650.pth --exp-config configs/dgcnn_simpleview_run_1.yaml

# evaluation on test set, pointnet++ protocol, table 4, row 2 (no vote)
python main.py --entry test --model-path pretrained/pointnet2_rscnn_run_1/model_625.pth --exp-config configs/pointnet2_rscnn_run_1.yaml
python main.py --entry test --model-path pretrained/pointnet2_dgcnn_run_1/model_400.pth --exp-config configs/pointnet2_dgcnn_run_1.yaml
python main.py --entry test --model-path pretrained/pointnet2_pointnet_run_1/model_350.pth --exp-config configs/pointnet2_pointnet_run_1.yaml
python main.py --entry test --model-path pretrained/pointnet2_pointnet2_run_1/model_925.pth --exp-config configs/pointnet2_pointnet2_run_1.yaml
python main.py --entry test --model-path pretrained/pointnet2_simpleview_run_1/model_625.pth --exp-config configs/pointnet2_simpleview_run_1.yaml

# evaluation on test set, pointnet++ protocol with vote, table 4, row 3 (vote)
# python main.py --entry pn2_vote --model-path pretrained/pointnet2_rscnn_run_1/model_625.pth --exp-config configs/pointnet2_rscnn_run_1.yaml
# python main.py --entry pn2_vote --model-path pretrained/pointnet2_dgcnn_run_1/model_400.pth --exp-config configs/pointnet2_dgcnn_run_1.yaml
# python main.py --entry pn2_vote --model-path pretrained/pointnet2_pointnet_run_1/model_350.pth --exp-config configs/pointnet2_pointnet_run_1.yaml
# python main.py --entry pn2_vote --model-path pretrained/pointnet2_pointnet2_run_1/model_925.pth --exp-config configs/pointnet2_pointnet2_run_1.yaml
# python main.py --entry pn2_vote --model-path pretrained/pointnet2_simpleview_run_1/model_625.pth --exp-config configs/pointnet2_simpleview_run_1.yaml

# evaluation on test set, rscnn protocol, table 4, row 4 (no vote)
python main.py --entry test --model-path pretrained/rscnn_rscnn_run_1/model_best_test.pth --exp-config configs/rscnn_rscnn_run_1.yaml
python main.py --entry test --model-path pretrained/rscnn_dgcnn_run_1/model_best_test.pth --exp-config configs/rscnn_dgcnn_run_1.yaml
python main.py --entry test --model-path pretrained/rscnn_pointnet_run_1/model_best_test.pth --exp-config configs/rscnn_pointnet_run_1.yaml
python main.py --entry test --model-path pretrained/rscnn_pointnet2_run_1/model_best_test.pth --exp-config configs/rscnn_pointnet2_run_1.yaml
python main.py --entry test --model-path pretrained/rscnn_simpleview_run_1/model_best_test.pth --exp-config configs/rscnn_simpleview_run_1.yaml

# evaluation on test set, rscnn protocol with vote, table 4, row 4 (vote)
# python main.py --entry rscnn_vote --model-path pretrained/rscnn_rscnn_run_1/model_best_test.pth --exp-config configs/cls_dl_rscnn_model_rscnn_run_1.yaml
# python main.py --entry rscnn_vote --model-path pretrained/rscnn_dgcnn_run_1/model_best_test.pth --exp-config configs/cls_dl_rscnn_model_dgcnn_run_1.yaml
# python main.py --entry rscnn_vote --model-path pretrained/rscnn_pointnet_run_1/model_best_test.pth --exp-config configs/cls_dl_rscnn_model_pointnet_run_1.yaml
# python main.py --entry rscnn_vote --model-path pretrained/rscnn_pointnet2_run_1/model_best_test.pth --exp-config configs/cls_dl_rscnn_model_pointnet2_run_1.yaml
# python main.py --entry rscnn_vote --model-path pretrained/rscnn_simpleview_run_1/model_best_test.pth --exp-config configs/cls_dl_rscnn_model_multiview2_18_run_1.yaml

# evaluation on test set, dgcnn protocol, table 4, row 5 (CE)
python main.py --entry test --model-path pretrained/dgcnn_rscnn_ce_run_1/model_best_test.pth --exp-config configs/dgcnn_rscnn_ce_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_dgcnn_ce_run_1/model_best_test.pth --exp-config configs/dgcnn_dgcnn_ce_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet_ce_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet_ce_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet2_ce_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet2_ce_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_simpleview_ce_run_1/model_best_test.pth --exp-config configs/dgcnn_simpleview_ce_run_1.yaml

# evaluation on test set, dgcnn protocol, table 4, row 6 (smooth)
python main.py --entry test --model-path pretrained/dgcnn_rscnn_run_1/model_best_test.pth --exp-config configs/dgcnn_rscnn_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_dgcnn_run_1/model_best_test.pth --exp-config configs/dgcnn_dgcnn_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_pointnet2_run_1/model_best_test.pth --exp-config configs/dgcnn_pointnet2_run_1.yaml
python main.py --entry test --model-path pretrained/dgcnn_simpleview_run_1/model_best_test.pth --exp-config configs/dgcnn_simpleview_run_1.yaml
