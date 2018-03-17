import sys
caffe_root = '/home/zkx/caffe/'
sys.path.insert(0, caffe_root + 'python')

from caffe import layers as L, params as P, to_proto
root = '/home/zkx/caffe/'
deploy = root + 'prd_un_cnn_13/deploy.prototxt'

def create_deploy():
  conv1 = L.Convolution(bottom = 'data', kernel_size = 5, stride = 1, num_output = 15, pad = 0, weight_filler = dict(type = 'xavier'))
  pool1 = L.Pooling(conv1, pool = P.Pooling.MAX, kernel_size = 2, stride = 2)
  fc3 = L.InnerProduct(pool1, num_output = 300, weight_filler = dict(type = 'xavier'))
  relu3 = L.ReLU(fc3, in_place = True)
  fc4 = L.InnerProduct(relu3, num_output = 13, weight_filler = dict(type='xavier'))
  prob = L.Softmax(fc4)
  return to_proto(prob)

def write_deploy():
  with open(deploy, 'w') as f:
    f.write('name:"Lenet"\n')
    f.write('input:"data"\n')
    f.write('input_dim:1\n')
    f.write('input_dim:3\n')
    f.write('input_dim:55\n')
    f.write('input_dim:55\n')
    f.write(str(create_deploy()))

if __name__ == '__main__':
  write_deploy()
