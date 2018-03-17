#!/usr/bin/env sh
echo "Create train lmdb.."
rm -rf img_train_lmdb
/home/zkx/caffe/build/tools/convert_imageset \
--shuffle \
--resize_height=55 \
--resize_width=55 \
/home/zkx/caffe/prd_un_cnn_13/trainset/ \
train.txt \
img_train_lmdb

echo "Create test lmdb.."
rm -rf img_test_lmdb
/home/zkx/caffe/build/tools/convert_imageset \
--shuffle \
--resize_width=55 \
--resize_height=55 \
/home/zkx/caffe/prd_un_cnn_13/testset/ \
test.txt \
img_test_lmdb

echo "All Done.."
